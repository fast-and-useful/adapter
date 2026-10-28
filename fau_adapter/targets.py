"""`fau-adapter targets`: the training pairs. For every prompt and noise, the unguided base runs its fine grid from
pure noise down to the jump endpoint (one of the student's own noise levels); the pair is the noise and the base's
latent there. The base is needed only here.

The pairs are written in parts of --part-size, so an interrupted run resumes where it stopped and several GPUs can
share the work (--worker i --num-workers n). A finished directory holds config.json, prompts.json and the parts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from . import __version__, prompts as P
from .models import MODELS, RESOLUTION, get_model


def part_name(p: int) -> str:
    return f"part-{p:05d}.safetensors"


def add_parser(sub) -> None:
    ap = sub.add_parser("targets", help="run the unguided base to make the adapter's training pairs",
                        description="Run the unguided base model from many noises and save its latent at the jump "
                                    "endpoint: the adapter's training pairs. The base is needed only here.")
    ap.add_argument("--model", required=True, choices=sorted(MODELS), help="the student to adapt")
    P.add_arguments(ap, per_group_default=160, per_group_help="noises per class (each class cycles through the eight "
                    "templates) or per prompt; the paper: 160 per class for the 100 ImageNet-100 classes, 16,000 pairs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--target-level", type=int, default=None, metavar="L",
                    help="the jump endpoint as an index into the student's own levels (default: the paper's choice)")
    ap.add_argument("--teacher-guidance", type=float, default=1.0, metavar="W",
                    help="the base's guidance w in neg + w (pos - neg); 1 (default) is unguided, which the paper's "
                         "ablations show matters")
    ap.add_argument("--part-size", type=int, default=500, help="pairs per output file (default 500)")
    ap.add_argument("--batch-size", type=int, default=None, help="images per teacher batch (default: the model's)")
    ap.add_argument("--worker", type=int, default=0, help="this process's share of the parts: parts p with "
                                                          "p %% num_workers == worker")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--teacher", default=None, help="a local copy (or mirror) of the base model")
    ap.add_argument("--student", default=None, help="a local copy (or mirror) of the student (its scheduler sets "
                                                    "the noise levels)")
    ap.set_defaults(run=run)


def run(args) -> None:
    model = get_model(args.model, args.student, args.teacher)
    prompts = P.from_arguments(args)
    level = model.default_target_level if args.target_level is None else args.target_level
    model.check_level(level)
    if not 0 <= args.worker < args.num_workers:
        raise SystemExit("--worker must lie in 0 .. --num-workers - 1")
    out = Path(args.out)
    jobs = [(k, j) for k in range(len(prompts)) for j in range(args.per_group)]
    parts = [jobs[i:i + args.part_size] for i in range(0, len(jobs), args.part_size)]
    s0, s1 = model.sigmas(level)
    config = dict(model=model.name, student=type(model).student, teacher=type(model).teacher, resolution=RESOLUTION,
                  target_level=level, target=model.student_levels()[level], sigma_start=s0, sigma_target=s1,
                  teacher_guidance=args.teacher_guidance, teacher_steps=len(model.teacher_grid(level)) - 1,
                  prompt_source=prompts.source, templated=prompts.templated, groups=len(prompts),
                  per_group=args.per_group, pairs=len(jobs), seed_offset=P.PAIR_SEED_OFFSET,
                  part_size=args.part_size, num_parts=len(parts), version=__version__)
    write_config(out, config, prompts)

    mine = [p for p in range(len(parts)) if p % args.num_workers == args.worker and not (out / part_name(p)).exists()]
    print(f"{model.name}: {len(jobs)} pairs of {len(prompts)} {'classes' if prompts.templated else 'prompts'}, "
          f"target {model.describe_level(level)} ({config['teacher_steps']} teacher steps, guidance "
          f"{args.teacher_guidance:g}); worker {args.worker}/{args.num_workers}: {len(mine)} parts to make", flush=True)
    if not mine:
        return
    pipe = model.load_teacher().to(args.device)
    bs = args.batch_size or model.batch_size
    t0, done = time.time(), 0
    for p in mine:
        rec = dict(noise=[], target=[])
        for b in range(0, len(parts[p]), bs):
            batch = parts[p][b:b + bs]
            x1, xt = model.teacher_states(pipe, [prompts.prompt(k, j) for k, j in batch],
                                          [P.seed(k, j, P.PAIR_SEED_OFFSET) for k, j in batch], level, args.teacher_guidance)
            rec["noise"].append(x1)
            rec["target"].append(xt)
        tensors = {name: torch.cat(v).contiguous() for name, v in rec.items()}
        tensors["prompt_id"] = torch.tensor([prompts.prompt_id(k, j) for k, j in parts[p]])
        tensors["group"] = torch.tensor([k for k, _ in parts[p]])
        tensors["index"] = torch.tensor([j for _, j in parts[p]])
        save_part(out / part_name(p), tensors)
        done += len(parts[p])
        each = (time.time() - t0) / done
        left = sum(len(parts[q]) for q in mine[mine.index(p) + 1:])
        print(f"part {p + 1}/{len(parts)}: {done} pairs in {time.time() - t0:.0f} s ({each:.2f} s each, "
              f"{left * each / 60:.0f} min left for this worker)", flush=True)
    print(f"done: {out}", flush=True)


def write_config(out: Path, config: dict, prompts: P.PromptSet) -> None:
    """Writes config.json and prompts.json, or checks that a resumed run asks for the same pairs."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), v) for k, v in config.items()   # prompts compare by content: a path may be relative
                if k not in ("version", "prompt_source") and old.get(k) != v}
        if json.loads((out / "prompts.json").read_text(encoding="utf-8"))["prompts"] != prompts.distinct():
            diff["prompts"] = ("other prompts", "these")
        if diff:
            raise SystemExit(f"{out} holds other targets ({', '.join(f'{k}: {a!r} there, {b!r} here' for k, (a, b) in diff.items())}); "
                             "choose another --out")
        return
    tmp = out / "prompts.json.tmp"
    tmp.write_text(json.dumps(dict(prompts=prompts.distinct(), labels=list(prompts.labels)), ensure_ascii=False))
    tmp.replace(out / "prompts.json")
    tmp = out / "config.json.tmp"
    tmp.write_text(json.dumps(config, indent=1))
    tmp.replace(path)


def save_part(path: Path, tensors: dict) -> None:
    from safetensors.torch import save_file
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(path)


class TargetSet:
    """A finished targets directory in memory: its configuration, its prompts, and every pair's noise, target and
    prompt."""

    def __init__(self, root: str | Path):
        from safetensors import safe_open
        root = Path(root)
        if not (root / "config.json").exists():
            raise SystemExit(f"{root}: no config.json; make the targets first (fau-adapter targets)")
        self.root = root
        self.config = json.loads((root / "config.json").read_text())
        self.prompts = json.loads((root / "prompts.json").read_text(encoding="utf-8"))["prompts"]
        paths = [root / part_name(p) for p in range(self.config["num_parts"])]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{root}: {len(missing)} of {len(paths)} parts are missing ({missing[0]}, ...); run the same "
                             "`fau-adapter targets` command again to finish them")
        n = self.config["pairs"]
        with safe_open(str(paths[0]), framework="pt") as f:
            first = {k: f.get_tensor(k) for k in ("noise", "target")}
        self.noise = torch.empty((n, *first["noise"].shape[1:]), dtype=first["noise"].dtype)
        self.target = torch.empty((n, *first["target"].shape[1:]), dtype=first["target"].dtype)
        self.prompt_id = torch.empty(n, dtype=torch.long)
        at = 0
        for path in paths:
            with safe_open(str(path), framework="pt") as f:
                m = f.get_slice("prompt_id").get_shape()[0]
                self.noise[at:at + m] = f.get_tensor("noise")
                self.target[at:at + m] = f.get_tensor("target")
                self.prompt_id[at:at + m] = f.get_tensor("prompt_id")
            at += m
        if at != n:
            raise SystemExit(f"{root}: the parts hold {at} pairs, config.json says {n}")

    def __len__(self):
        return len(self.prompt_id)
