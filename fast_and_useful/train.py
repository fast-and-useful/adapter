"""`fast-and-useful train`: fit the adapter. Low-rank updates on the student's denoiser are trained so that its first
prediction, one Euler step from pure noise, lands on the teacher's latent at the jump endpoint:

    loss = mean || x1 + (s_a - s_1) v(x1, s_1, prompt) - x*(x1, prompt) ||^2

with the released weights frozen and B zero at the start. A held-out 5% of the pairs (at least 64) measures how far the
first prediction is from its target, relative to how far the noise is (relative error) and to how far an empty image's
latent is (content error, which compares across models); 1 is no better than not moving.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from . import __version__, lora
from .models import get_model
from .targets import TargetSet


def add_parser(sub) -> None:
    ap = sub.add_parser("train", help="fit the adapter on the pairs made by `targets`",
                        description="Fit low-rank updates on the student's denoiser so that its first prediction "
                                    "jumps to the teacher's latent at the jump endpoint.")
    ap.add_argument("--targets", required=True, help="the directory written by `fast-and-useful targets`")
    ap.add_argument("--out", required=True, help="output directory: adapter.safetensors and training.json")
    ap.add_argument("--rank", type=int, default=32, help="rank of the update (default 32, the paper's)")
    ap.add_argument("--steps", type=int, default=3000, help="optimization steps (default 3000)")
    ap.add_argument("--batch-size", type=int, default=None, help="pairs per step (default: the paper's, 8 or 16)")
    ap.add_argument("--grad-accum", type=int, default=1, help="split each step's batch into this many passes to save "
                                                              "GPU memory (same result, slower)")
    ap.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default 1e-4)")
    ap.add_argument("--warmup", type=int, default=100, help="linear warmup steps before the cosine decay")
    ap.add_argument("--seed", type=int, default=0, help="seeds the initialization and the batch order")
    ap.add_argument("--eval-every", type=int, default=500, help="steps between held-out evaluations")
    ap.add_argument("--embedding-cache-gb", type=float, default=16.0,
                    help="keep the prompt embeddings in CPU memory up to this size; beyond it, the text encoder stays on "
                         "the GPU and encodes every batch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--student", default=None, help="a local copy (or mirror) of the student")
    ap.set_defaults(run=run)


class PromptEncoder:
    """Prompt embeddings by prompt id: computed once and kept in CPU memory when they fit, else per batch."""

    def __init__(self, model, pipe, prompts: list[str], device, max_gb: float):
        self.model, self.pipe, self.prompts, self.device = model, pipe, prompts, device
        for te in model.text_encoders(pipe):
            te.to(device)
        first = model.encode(pipe, prompts[:1], device)[0]
        size = first.numel() * first.element_size() * len(prompts) / 1e9
        if size > max_gb:
            self.cache = None
            print(f"{len(prompts)} prompts: their embeddings would take {size:.1f} GB, so each batch is encoded on the GPU",
                  flush=True)
            return
        self.cache = []
        for i in range(0, len(prompts), 16):
            self.cache += [e.cpu() for e in model.encode(pipe, prompts[i:i + 16], device)]
        for te in model.text_encoders(pipe):
            te.to("cpu")
        torch.cuda.empty_cache()
        print(f"{len(prompts)} prompt embeddings in memory ({size:.2f} GB)", flush=True)

    def __call__(self, ids):
        ids = [int(i) for i in ids]
        if self.cache is None:
            embs = self.model.encode(self.pipe, [self.prompts[i] for i in ids], self.device)
        else:
            embs = [self.cache[i] for i in ids]
        return self.model.collate(embs, self.device)


def split(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(held-out, training) pair indices: a fixed 5% (at least 64, at most half) held out."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    k = min(max(64, n // 20), n // 2)
    return perm[:k], perm[k:]


def run(args) -> None:
    data = TargetSet(args.targets)
    cfg = data.config
    model = get_model(cfg["model"], args.student)
    level = cfg["target_level"]
    device = torch.device(args.device)
    s0, s1 = model.sigmas(level)
    if (round(s0, 4), round(s1, 4)) != (round(cfg["sigma_start"], 4), round(cfg["sigma_target"], 4)):
        raise SystemExit(f"{args.targets} was made with noise levels {cfg['sigma_start']} -> {cfg['sigma_target']}; "
                         f"{model.student} gives {s0} -> {s1}")
    bs = args.batch_size or model.train_batch_size
    if bs % args.grad_accum:
        raise SystemExit("--batch-size must be a multiple of --grad-accum")
    if args.steps < 1 or args.warmup < 1:
        raise SystemExit("--steps and --warmup must be at least 1")
    torch.manual_seed(args.seed)
    held, train_ids = split(len(data))
    print(f"{model.name}: {len(data)} pairs ({len(train_ids)} for training, {len(held)} held out), target "
          f"{model.describe_level(level)}, teacher guidance {cfg['teacher_guidance']:g}", flush=True)

    pipe = model.load_student()
    encode = PromptEncoder(model, pipe, data.prompts, device, args.embedding_cache_gb)
    net = model.training_denoiser(pipe, device)
    layers = lora.attach(net, model.lora_layers(net), args.rank)
    for layer in layers.values():
        layer.merge_dtype = model.dtype   # "merged" emulates generation's precision
    params = lora.parameters(layers)
    print(f"low-rank updates on {len(layers)} linear layers: {sum(p.numel() for p in params) / 1e6:.1f}M parameters",
          flush=True)
    step = model.first_step(pipe, net, level)

    def predict(ids, inference=False):
        return step(data.noise[ids].to(device), encode(data.prompt_id[ids]), inference)

    def evaluate(inference=False):
        net.eval()
        rel, con = [], []
        with torch.no_grad():
            for i in range(0, len(held), bs):
                ids = held[i:i + bs]
                pred = predict(ids, inference)
                x1, xt = data.noise[ids].to(device).float(), data.target[ids].to(device).float()
                err = (pred - xt).pow(2).flatten(1).sum(1)
                rel += (err / (x1 - xt).pow(2).flatten(1).sum(1)).tolist()
                con += (err / (x1 * (s1 / s0) - xt).pow(2).flatten(1).sum(1)).tolist()
        net.train()
        return sum(rel) / len(rel), sum(con) / len(con)

    with torch.no_grad():   # B starts at zero: the adapter must leave the released model exactly as it is
        on = predict(held[:2])
        lora.set_mode(layers, "off")
        off = predict(held[:2])
        lora.set_mode(layers, "lowrank")
    if not torch.equal(on, off):
        raise RuntimeError("the zero-initialized adapter changes the denoiser's output")
    log = dict(start=evaluate(), curve=[])
    print(f"held-out error of the released first step: relative {log['start'][0]:.4f}, content {log['start'][1]:.4f}",
          flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    net.train()
    t0 = time.time()
    for s in range(args.steps):
        ids = train_ids[torch.randint(len(train_ids), (bs,))]
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for chunk in ids.chunk(args.grad_accum):
            loss = (predict(chunk) - data.target[chunk].to(device).float()).pow(2).mean() * (len(chunk) / bs)
            loss.backward()
            total += loss.item()
        opt.step()
        sched.step()
        if (s + 1) % 50 == 0:
            each = (time.time() - t0) / (s + 1)
            print(f"step {s + 1}/{args.steps}: loss {total:.5f}, {each:.2f} s/step, "
                  f"{(args.steps - s - 1) * each / 60:.0f} min left", flush=True)
        if (s + 1) % args.eval_every == 0 or s + 1 == args.steps:
            rel, con = evaluate()
            log["curve"].append((s + 1, rel, con))
            print(f"step {s + 1}: held-out error relative {rel:.4f}, content {con:.4f}", flush=True)
    minutes = (time.time() - t0) / 60

    lora.set_mode(layers, "merged")   # the update folded into the weights in generation's precision
    log["merged"] = evaluate(inference=True)
    lora.set_mode(layers, "lowrank")
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
    print(f"as generation runs it (merged): held-out error relative {log['merged'][0]:.4f}, content "
          f"{log['merged'][1]:.4f}; {minutes:.0f} min, peak GPU memory {peak:.0f} GB", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(model=model.name, student=type(model).student, teacher=cfg["teacher"], resolution=cfg["resolution"],
                  target_level=level, target=cfg["target"], teacher_guidance=cfg["teacher_guidance"], rank=args.rank,
                  steps=args.steps, batch_size=bs, lr=args.lr, warmup=args.warmup, seed=args.seed,
                  pairs=len(data), prompt_source=cfg["prompt_source"], targets=str(Path(args.targets).resolve()),
                  held_out_error=dict(released=log["start"], adapted=log["curve"][-1][1:], merged=log["merged"]),
                  version=__version__)
    lora.save(out / "adapter.safetensors", layers, model.denoiser_name, config)
    (out / "training.json").write_text(json.dumps(dict(config=config, log=log, minutes=minutes, peak_gb=peak), indent=1))
    print(f"saved {out / 'adapter.safetensors'}", flush=True)
