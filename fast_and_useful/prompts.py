"""Prompts and seeds.

A prompt set is a list of groups. A group is either a class name, prompted through eight photo templates (image j of a
class uses template j mod 8), or one free-form prompt (a caption) from a file. Image j of group k has the seed
1,000,003 k + j; the adapter's training pairs add 600,000, so they never share a noise with a generated dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

TEMPLATES = ("a photo of a {}.", "a close-up photo of a {}.", "a photo of the {} in its natural setting.",
             "a bright photo of a {}.", "a photo of a small {}.", "a photo of a large {}.", "a cropped photo of a {}.",
             "a good photo of a {}.")
PAIR_SEED_OFFSET = 600_000
BUILTIN_CLASSES = ("imagenet100", "imagenet1k")


def seed(group: int, index: int, offset: int = 0) -> int:
    return 1_000_003 * group + index + offset


@dataclass(frozen=True)
class PromptSet:
    labels: tuple[str, ...]   # one per group: a class id (the folder name of its images) or the prompt's number
    texts: tuple[str, ...]    # one per group: a class name or a prompt
    templated: bool           # class names go through TEMPLATES, prompts are used as they are
    source: str               # where the groups came from, for the records

    def __len__(self):
        return len(self.texts)

    @property
    def per_group(self) -> int:
        """Distinct prompts per group."""
        return len(TEMPLATES) if self.templated else 1

    def prompt(self, group: int, index: int) -> str:
        if self.templated:
            return TEMPLATES[index % len(TEMPLATES)].format(self.texts[group])
        return self.texts[group]

    def prompt_id(self, group: int, index: int) -> int:
        """Index of prompt(group, index) in distinct()."""
        return group * self.per_group + index % self.per_group

    def distinct(self) -> list[str]:
        return [self.prompt(g, i) for g in range(len(self)) for i in range(self.per_group)]

    @classmethod
    def from_classes(cls, spec: str) -> "PromptSet":
        """A built-in class list (imagenet100, imagenet1k) or a text file with one class per line, either
        `<id><TAB><name>` (the id names the class's image folder) or `<name>`. Blank lines and lines starting
        with # are skipped."""
        if spec in BUILTIN_CLASSES:
            text = resources.files("fast_and_useful").joinpath("classes", f"{spec}.txt").read_text(encoding="utf-8")
        else:
            text = Path(spec).read_text(encoding="utf-8")
        labels, names = [], []
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            label, _, name = line.rstrip("\n").partition("\t")
            name = name.strip() if name else label.strip()
            labels.append(label.strip().replace("/", "_"))
            names.append(name)
        if not names:
            raise ValueError(f"{spec}: no classes")
        for what, values in (("names", names), ("ids", labels)):
            dup = sorted({v for v in values if values.count(v) > 1})
            if dup:
                raise ValueError(f"{spec}: repeated class {what} {dup}: every class needs its own")
        return cls(tuple(labels), tuple(names), True, spec)

    @classmethod
    def from_file(cls, path: str) -> "PromptSet":
        """One prompt per line (blank lines skipped), used as it is."""
        texts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not texts:
            raise ValueError(f"{path}: no prompts")
        return cls(tuple(f"{k:06d}" for k in range(len(texts))), tuple(texts), False, path)

    @classmethod
    def from_text(cls, text: str) -> "PromptSet":
        return cls(("000000",), (text,), False, "--prompt")


def add_arguments(parser, per_group_default: int, per_group_help: str) -> None:
    """--classes / --prompts / --prompt and how many items per group."""
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--classes", help="class names, each prompted through eight photo templates: imagenet100, "
                                       "imagenet1k, or a text file with one class per line ('<id><TAB><name>' or '<name>')")
    src.add_argument("--prompts", help="a text file with one prompt per line, used as it is (e.g. captions)")
    src.add_argument("--prompt", help="one prompt")
    parser.add_argument("--per-class", "--per-prompt", dest="per_group", type=int, default=per_group_default, metavar="N",
                        help=per_group_help)


def from_arguments(args) -> PromptSet:
    if args.classes:
        return PromptSet.from_classes(args.classes)
    if args.prompts:
        return PromptSet.from_file(args.prompts)
    return PromptSet.from_text(args.prompt)
