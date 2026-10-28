"""Low-rank updates on a frozen denoiser's linear layers: W becomes W + B A, with B zero at the start, so training starts
from the released model exactly. A wrapped layer runs in one of three modes:

    "lowrank"  W x + B (A x), the low-rank branch in float32 (training)
    "merged"   (W + B A) x, the update folded into W in the denoiser's inference precision (generation; the paper's
               images were made this way)
    "off"      W x, the released model (every step after the first)

An adapter is saved as safetensors: `<denoiser>.<layer>.lora_A.weight` (rank x in) and `<denoiser>.<layer>.lora_B.weight`
(out x rank) in float32, the update's scale fixed at 1, and the training configuration in the file's metadata.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

FORMAT = "fau-adapter"
MODES = ("lowrank", "merged", "off")


class LoRALinear(nn.Module):
    """A linear layer with a low-rank update B A beside it."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        device = base.weight.device
        self.base = base
        # A child registered after `base`: a wrapped model's first parameter stays a released weight, the one diffusers
        # reads the model's dtype from
        self.update = nn.ParameterDict({
            "A": nn.Parameter(torch.randn(rank, base.in_features, device=device) / math.sqrt(base.in_features)),
            "B": nn.Parameter(torch.zeros(base.out_features, rank, device=device))})
        self.mode = "lowrank"
        self.merge_dtype = base.weight.dtype   # the precision W + B A is rounded to in "merged" mode

    @property
    def A(self) -> nn.Parameter:
        return self.update["A"]

    @property
    def B(self) -> nn.Parameter:
        return self.update["B"]

    def forward(self, x):
        if self.mode == "off":
            return self.base(x)
        if self.mode == "merged":
            w = self.base.weight
            delta = (self.B.float() @ self.A.float()).to(self.merge_dtype)
            return F.linear(x, (w.to(self.merge_dtype) + delta).to(w.dtype), self.base.bias)
        y = self.base(x)
        return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype)


def attach(net: nn.Module, names: list[str], rank: int) -> dict[str, LoRALinear]:
    """Wraps the named linear layers of `net` in LoRALinear (in place) and returns them by name."""
    layers = {}
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = net.get_submodule(parent_name) if parent_name else net
        base = getattr(parent, child)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is a {type(base).__name__}, not a linear layer")
        layers[name] = LoRALinear(base, rank)
        setattr(parent, child, layers[name])
    return layers


def set_mode(layers: dict[str, LoRALinear], mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    for layer in layers.values():
        layer.mode = mode


def parameters(layers: dict[str, LoRALinear]) -> list[nn.Parameter]:
    return [p for layer in layers.values() for p in (layer.A, layer.B)]


def save(path: str | Path, layers: dict[str, LoRALinear], prefix: str, config: dict) -> None:
    """Writes the adapter and its configuration (`config`, any JSON-serializable dict) to one safetensors file."""
    from safetensors.torch import save_file
    tensors = {}
    for name, layer in layers.items():
        tensors[f"{prefix}.{name}.lora_A.weight"] = layer.A.detach().float().cpu().contiguous()
        tensors[f"{prefix}.{name}.lora_B.weight"] = layer.B.detach().float().cpu().contiguous()
    rank = next(iter(layers.values())).A.shape[0]
    metadata = {"format": FORMAT, "version": "1", "denoiser": prefix, "rank": str(rank), "config": json.dumps(config)}
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata=metadata)
    tmp.replace(path)


def load(path: str | Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """-> ({layer name: (A, B)}, configuration) of an adapter file written by save()."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(f"{path} is not an adapter written by fau-adapter (metadata: {metadata})")
        prefix = metadata["denoiser"] + "."
        weights = {}
        for key in f.keys():
            name, kind, _ = key[len(prefix):].rsplit(".", 2)   # <layer>.lora_A.weight
            weights.setdefault(name, {})[kind] = f.get_tensor(key)
    config = json.loads(metadata["config"])
    return {name: (w["lora_A"], w["lora_B"]) for name, w in weights.items()}, config


def attach_saved(net: nn.Module, weights: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, LoRALinear]:
    """Wraps the layers an adapter file names and loads its weights (frozen, for generation)."""
    rank = next(iter(weights.values()))[0].shape[0]
    layers = attach(net, list(weights), rank)
    with torch.no_grad():
        for name, (A, B) in weights.items():
            layer = layers[name]
            if A.shape != layer.A.shape or B.shape != layer.B.shape:
                raise ValueError(f"{name}: the adapter's shapes {tuple(A.shape)}, {tuple(B.shape)} do not fit the layer")
            layer.A.copy_(A)
            layer.B.copy_(B)
            layer.A.requires_grad_(False)
            layer.B.requires_grad_(False)
    return layers
