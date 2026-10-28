import torch
import torch.nn.functional as F
from torch import nn

from fau_adapter import lora


def toy(dtype=torch.float32):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)).to(dtype)


def trained(layers):
    """Gives every update a nonzero B, as training would."""
    with torch.no_grad():
        for layer in layers.values():
            layer.B.normal_(std=0.05)


def test_zero_initialized_update_keeps_the_released_model():
    net, x = toy(), torch.randn(4, 16)
    ref = net(x)
    lora.attach(net, ["0", "2"], rank=4)
    assert torch.equal(net(x), ref)


def test_off_mode_is_the_released_model():
    net, x = toy(), torch.randn(4, 16)
    ref = net(x)
    layers = lora.attach(net, ["0", "2"], rank=4)
    trained(layers)
    assert not torch.allclose(net(x), ref)
    lora.set_mode(layers, "off")
    assert torch.equal(net(x), ref)


def test_merged_mode_folds_the_update_into_the_weights_in_their_precision():
    for dtype in (torch.float32, torch.bfloat16):
        base = nn.Linear(16, 8).to(dtype)
        layer = lora.LoRALinear(base, rank=4)
        with torch.no_grad():
            layer.B.normal_(std=0.05)
        layer.mode = "merged"
        x = torch.randn(3, 16).to(dtype)
        folded = base.weight + (layer.B.float() @ layer.A.float()).to(dtype)
        assert torch.equal(layer(x), F.linear(x, folded, base.bias))


def test_lowrank_and_merged_agree_in_float32():
    net, x = toy(), torch.randn(4, 16)
    layers = lora.attach(net, ["0", "2"], rank=4)
    trained(layers)
    low = net(x)
    lora.set_mode(layers, "merged")
    assert torch.allclose(net(x), low, atol=1e-5)


def test_the_first_parameter_stays_a_released_weight():
    """diffusers reads a model's dtype from its first parameter; the float32 update must not come first."""
    net = toy(torch.bfloat16)
    lora.attach(net, ["0", "2"], rank=4)
    name, first = next(iter(net.named_parameters()))
    assert first.dtype == torch.bfloat16 and name == "0.base.weight"


def test_save_and_load_round_trip(tmp_path):
    net, x = toy(), torch.randn(4, 16)
    layers = lora.attach(net, ["0", "2"], rank=4)
    trained(layers)
    lora.set_mode(layers, "merged")
    ref = net(x)
    path = tmp_path / "adapter.safetensors"
    lora.save(path, layers, "transformer", {"model": "toy", "target_level": 3})

    weights, config = lora.load(path)
    assert config == {"model": "toy", "target_level": 3}
    assert sorted(weights) == ["0", "2"]
    fresh = toy()
    loaded = lora.attach_saved(fresh, weights)
    lora.set_mode(loaded, "merged")
    assert torch.equal(fresh(x), ref)
    assert not any(p.requires_grad for layer in loaded.values() for p in (layer.A, layer.B))


def test_keys_follow_the_lora_naming(tmp_path):
    from safetensors import safe_open
    net = toy()
    layers = lora.attach(net, ["0"], rank=4)
    path = tmp_path / "adapter.safetensors"
    lora.save(path, layers, "unet", {})
    with safe_open(str(path), framework="pt") as f:
        assert sorted(f.keys()) == ["unet.0.lora_A.weight", "unet.0.lora_B.weight"]
        assert f.get_tensor("unet.0.lora_A.weight").shape == (4, 16)
        assert f.get_tensor("unet.0.lora_B.weight").shape == (32, 4)
        assert f.metadata()["rank"] == "4"
