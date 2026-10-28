"""The schedules against the noise levels the paper's adapters were trained on. SD-Turbo's and Z-Image's tests read the
models' scheduler configurations from the Hugging Face Hub (a few kB each) and are skipped without it."""
import pytest

from fau_adapter.models import MODELS, Flux2Klein, SDTurbo, ZImageTurbo, get_model


def hub(model):
    try:
        model.student_levels() if isinstance(model, ZImageTurbo) else model.sigmas(1)
    except OSError as e:
        pytest.skip(f"scheduler configuration not available: {e}")
    return model


def test_registry():
    assert sorted(MODELS) == ["flux2-klein", "sd-turbo", "z-image-turbo"]
    with pytest.raises(ValueError):
        get_model("sdxl-turbo")


def test_klein_levels_at_512_px():
    m = Flux2Klein()
    assert m.student_levels() == [1.0, 0.958085371628187, 0.8839818317017591, 0.7174965587003479]
    assert m.default_target_level == 3 and m.target_levels() == [1, 2, 3]


def test_klein_teacher_grid():
    m = Flux2Klein()
    grid = m.teacher_grid(3)
    assert len(grid) == 36 and grid[0] == 1.0   # 35 teacher steps
    levels = m.student_levels()
    assert (grid[10], grid[22], grid[35]) == tuple(levels[1:])
    assert [round(v, 5) for v in (grid[9], grid[11], grid[21], grid[23], grid[34])] == \
        [0.96152, 0.95641, 0.89162, 0.87469, 0.73859]
    assert all(a > b for a, b in zip(grid, grid[1:]))
    assert m.teacher_grid(1) == grid[:11]


def test_klein_schedules():
    m = Flux2Klein()
    assert m.schedule(3) == [1.0, 0.7174965587003479]   # two denoiser evaluations
    assert m.schedule(None) == m.student_levels()       # the released student: four
    with pytest.raises(ValueError):
        m.schedule(0)


def test_sd_turbo_grid_and_noise_scales():
    m = hub(SDTurbo())
    grid = m.teacher_grid(2)
    assert grid[0] == 999 and grid[15] == 749 and grid[-1] == 499 and len(grid) == 31
    assert m.teacher_grid(3)[-1] == 249 and len(m.teacher_grid(3)) == 46
    assert m.schedule(2) == [999, 499, 249]   # three denoiser evaluations
    s0, s1 = m.sigmas(2)
    assert round(s0, 4) == 14.6146 and s1 == pytest.approx(1.6128870248794556, abs=1e-6)
    assert m.sigmas(1)[1] == pytest.approx(4.081730842590332, abs=1e-6)


def test_z_image_turbo_levels_and_grid():
    m = hub(ZImageTurbo())
    levels = m.student_levels()
    assert len(levels) == 9 and levels[0] == 1.0
    assert levels[1:4] == [0.9600000381469727, 0.9130435585975647, 0.8571428060531616]
    grid = m.teacher_grid(3)
    assert len(grid) == 27 and grid[-1] == levels[3]   # 26 teacher steps
    assert len(m.schedule(3)) == 7                      # seven denoiser evaluations
    assert m.target_levels() == [1, 2, 3]
