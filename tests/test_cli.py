import argparse
import json

import pytest
import torch

from fast_and_useful import __main__ as cli
from fast_and_useful import prompts as P
from fast_and_useful.targets import TargetSet, part_name, save_part, write_config


@pytest.mark.parametrize("command", ["targets", "train", "generate"])
def test_help(command, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([command, "--help"])
    assert e.value.code == 0
    assert "--out" in capsys.readouterr().out


def test_per_class_and_per_prompt_are_one_option():
    ap = argparse.ArgumentParser()
    P.add_arguments(ap, 160, "")
    assert ap.parse_args(["--classes", "imagenet100", "--per-class", "3"]).per_group == 3
    assert ap.parse_args(["--prompt", "a hare", "--per-prompt", "4"]).per_group == 4
    with pytest.raises(SystemExit):
        ap.parse_args(["--classes", "imagenet100", "--prompt", "a hare"])


def fake_targets(root, parts=2, per_part=3):
    prompts = P.PromptSet(("a", "b"), ("red panda", "hare"), True, "test")
    config = dict(model="flux2-klein", pairs=parts * per_part, num_parts=parts, target_level=3)
    write_config(root, config, prompts)
    for p in range(parts):
        n = per_part
        save_part(root / part_name(p), dict(noise=torch.full((n, 4, 2), float(p)), target=torch.zeros(n, 4, 2),
                                            prompt_id=torch.arange(n) + p, group=torch.zeros(n, dtype=torch.long),
                                            index=torch.arange(n)))
    return config, prompts


def test_target_set_reads_every_part(tmp_path):
    fake_targets(tmp_path)
    data = TargetSet(tmp_path)
    assert len(data) == 6 and data.noise.shape == (6, 4, 2)
    assert data.noise[:3].eq(0).all() and data.noise[3:].eq(1).all()
    assert data.prompt_id.tolist() == [0, 1, 2, 1, 2, 3]
    assert data.prompts == P.PromptSet(("a", "b"), ("red panda", "hare"), True, "test").distinct()


def test_target_set_refuses_missing_parts(tmp_path):
    fake_targets(tmp_path)
    (tmp_path / part_name(1)).unlink()
    with pytest.raises(SystemExit, match="missing"):
        TargetSet(tmp_path)


def test_a_resumed_run_must_ask_for_the_same_targets(tmp_path):
    config, prompts = fake_targets(tmp_path)
    write_config(tmp_path, config, prompts)   # same: fine
    with pytest.raises(SystemExit, match="other targets"):
        write_config(tmp_path, dict(config, target_level=2), prompts)
    assert json.loads((tmp_path / "config.json").read_text())["target_level"] == 3


def test_held_out_split():
    from fast_and_useful.train import split
    held, train = split(16000)
    assert len(held) == 800 and len(train) == 15200   # the paper's 5%
    assert torch.equal(held, split(16000)[0])         # fixed by its seed
    held, train = split(8)
    assert len(held) == 4 and len(train) == 4
    assert not set(held.tolist()) & set(train.tolist())
