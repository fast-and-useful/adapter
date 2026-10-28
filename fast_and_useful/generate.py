"""`fast-and-useful generate`: images from the adapted student. The adapter is on for the first prediction only: one
jump from pure noise to the jump endpoint; then it is switched off and the released student finishes on its own
remaining levels. The base model is not loaded. Without --adapter, the released student runs its own schedule (the
comparison).

Class names make an image folder per class (out/<class id>/<index>.jpg), ready for torchvision's ImageFolder; prompts
make out/<prompt number>_<index>.jpg. Image j of group k has the seed 1,000,003 k + j (+ --seed-offset), so the same
command gives the same images, an interrupted run resumes, and several GPUs can share the work (--worker/--num-workers).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from . import __version__, lora, prompts as P
from .models import MODELS, RESOLUTION, get_model


class AdaptedGenerator:
    """The released student with an adapter on its first prediction.

        generator = AdaptedGenerator("runs/flux2-klein/adapter")              # or an adapter.safetensors file
        images = generator(["a photo of a red panda."], seeds=[0])            # a list of PIL images

    AdaptedGenerator(model="flux2-klein") is the released student on its own schedule, for comparison.
    """

    def __init__(self, adapter: str | Path | None = None, model: str | None = None, device: str = "cuda",
                 student: str | None = None):
        if adapter is not None:
            path = Path(adapter)
            if path.is_dir():
                path = path / "adapter.safetensors"
            weights, self.config = lora.load(path)
            if model is not None and model != self.config["model"]:
                raise ValueError(f"{path} adapts {self.config['model']}, not {model}")
            self.model = get_model(self.config["model"], student)
            self.level = self.config["target_level"]
        elif model is not None:
            weights, self.config, self.level = None, None, None
            self.model = get_model(model, student)
        else:
            raise ValueError("give an adapter, or a model to run the released student")
        self.pipe = self.model.load_student().to(device)
        self.layers = {}
        if weights is not None:
            self.layers = lora.attach_saved(self.model.denoiser(self.pipe), weights)
        self.schedule = self.model.schedule(self.level)

    @property
    def nfe(self) -> int:
        """Denoiser evaluations per image."""
        return len(self.schedule)

    def _first_step_only(self, pipe, i, t, kwargs):
        if i == 0:
            lora.set_mode(self.layers, "off")
        return kwargs

    @torch.no_grad()
    def __call__(self, prompts: list[str], seeds: list[int]):
        if len(prompts) != len(seeds):
            raise ValueError("one seed per prompt")
        if not self.layers:
            return self.model.generate(self.pipe, prompts, seeds, self.schedule)
        lora.set_mode(self.layers, "merged")   # the update folded into the weights, as the paper generated
        try:
            return self.model.generate(self.pipe, prompts, seeds, self.schedule, callback=self._first_step_only)
        finally:
            lora.set_mode(self.layers, "off")


def add_parser(sub) -> None:
    ap = sub.add_parser("generate", help="generate images (or a class-folder dataset) with the adapted student",
                        description="Generate images with the adapted student: the adapter on the first prediction "
                                    "only, then the released student. Without --adapter, the released student alone.")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--adapter", help="adapter.safetensors, or the directory `fast-and-useful train` wrote")
    which.add_argument("--model", choices=sorted(MODELS), help="no adapter: the released student on its own schedule")
    P.add_arguments(ap, per_group_default=1, per_group_help="images per class or per prompt (default 1; the paper's "
                    "datasets: 500 per class for ImageNet-100, 1,281 for ImageNet-1k)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to every seed: another draw of the same set")
    ap.add_argument("--resize", type=int, default=None, metavar="PX",
                    help=f"resize so the short side is PX (images are made at {RESOLUTION} px; the paper stored "
                         "256 px copies for classifier training)")
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90)")
    ap.add_argument("--batch-size", type=int, default=None, help="images per batch (default: the model's)")
    ap.add_argument("--worker", type=int, default=0, help="this process's share: images i with "
                                                          "i %% num_workers == worker")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--student", default=None, help="a local copy (or mirror) of the student")
    ap.set_defaults(run=run)


def image_path(out: Path, prompts: P.PromptSet, group: int, index: int, ext: str) -> Path:
    if prompts.templated:
        return out / prompts.labels[group] / f"{index:05d}.{ext}"
    return out / f"{prompts.labels[group]}_{index:05d}.{ext}"


def save_image(img, path: Path, resize: int | None, quality: int) -> None:
    from PIL import Image
    img = img.convert("RGB")
    if resize and min(img.size) != resize:
        s = resize / min(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, format="JPEG" if path.suffix == ".jpg" else "PNG", quality=quality)
    tmp.replace(path)


def run(args) -> None:
    if not 0 <= args.worker < args.num_workers:
        raise SystemExit("--worker must lie in 0 .. --num-workers - 1")
    prompts = P.from_arguments(args)
    out = Path(args.out)
    jobs = [(k, j) for k in range(len(prompts)) for j in range(args.per_group)][args.worker::args.num_workers]
    todo = [(k, j) for k, j in jobs if not image_path(out, prompts, k, j, args.format).exists()]
    gen = AdaptedGenerator(args.adapter, args.model, args.device, args.student) if todo else None
    if gen is not None:
        record = dict(model=gen.model.name, student=gen.model.student, adapter=str(Path(args.adapter).resolve()) if
                      args.adapter else None, target_level=gen.level, schedule=gen.schedule, nfe=gen.nfe,
                      resolution=RESOLUTION, prompt_source=prompts.source, templated=prompts.templated,
                      groups=len(prompts), per_group=args.per_group, seed_offset=args.seed_offset, resize=args.resize,
                      version=__version__)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / f"generation.json.{args.worker}.tmp"
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(out / "generation.json")
        if not prompts.templated:
            tmp = out / f"prompts.txt.{args.worker}.tmp"
            tmp.write_text("".join(t + "\n" for t in prompts.texts), encoding="utf-8")
            tmp.replace(out / "prompts.txt")
    print(f"{len(jobs)} images for worker {args.worker}/{args.num_workers}, {len(jobs) - len(todo)} already there"
          + (f"; {gen.model.name} {'with its adapter' if gen.layers else 'as released'}, {gen.nfe} denoiser "
             f"evaluations per image" if gen else ""), flush=True)
    bs = args.batch_size or (gen.model.batch_size if gen else 1)
    t0 = time.time()
    for b in range(0, len(todo), bs):
        batch = todo[b:b + bs]
        images = gen([prompts.prompt(k, j) for k, j in batch], [P.seed(k, j, args.seed_offset) for k, j in batch])
        for (k, j), img in zip(batch, images):
            save_image(img, image_path(out, prompts, k, j, args.format), args.resize, args.quality)
        done = b + len(batch)
        if (b // bs) % 50 == 0 or done == len(todo):
            each = (time.time() - t0) / done
            print(f"{done}/{len(todo)} images, {each:.3f} s each, {(len(todo) - done) * each / 60:.0f} min left",
                  flush=True)
    print(f"done: {out}", flush=True)
