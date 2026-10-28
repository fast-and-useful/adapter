"""The released students the paper adapts, each with the base model that supplies its training targets.

A model knows the student's own noise levels and the teacher's fine grid, how to load both, the starting noise of an
image, how the unguided teacher reaches its target, and the first prediction the adapter trains: one Euler step from
pure noise straight to one of the student's own levels (the jump endpoint, given as an index into those levels). The
rest (prompts, files, training, datasets) is shared.

Images are 512 px, as in the paper. FLUX.2-klein's noise levels depend on the resolution, so an adapter belongs to the
resolution it was trained at. To add a generator, subclass Model and register it in MODELS.
"""
from __future__ import annotations

import functools
from typing import Callable

import numpy as np
import torch
from torch import nn

RESOLUTION = 512


class Model:
    """A released distilled student and the base model it was distilled from."""

    name: str                    # the --model name
    student: str                 # Hugging Face repository (or local copy) of the released student
    teacher: str                 # ... of its base model, needed for the targets only
    student_steps: int           # length of the student's own schedule
    default_target_level: int    # the paper's jump endpoint, an index into student_levels()
    dtype: torch.dtype           # the denoiser's precision at generation
    batch_size: int              # images per batch in `targets` and `generate` (the paper's)
    train_batch_size: int        # pairs per optimization step (the paper's)
    denoiser_name = "transformer"

    def __init__(self, student: str | None = None, teacher: str | None = None):
        if student:
            self.student = student
        if teacher:
            self.teacher = teacher

    # --- schedules -----------------------------------------------------------------------------------------------
    def student_levels(self) -> list:
        """The student's own schedule from pure noise down, without the final 0 (noise levels, or SD timesteps)."""
        raise NotImplementedError

    def fine_grid(self) -> list:
        """The teacher's fine schedule: the base's own levels with the student's levels put in."""
        raise NotImplementedError

    def target_levels(self) -> list[int]:
        """The student levels the adapter can jump to: those the teacher's grid passes through."""
        levels, grid = self.student_levels(), self.fine_grid()
        return [i for i in range(1, len(levels)) if levels[i] in grid]

    def check_level(self, level: int) -> None:
        if level not in self.target_levels():
            raise ValueError(f"{self.name}: the jump endpoint must be one of the student's levels {self.target_levels()} "
                             f"(its levels: {self.student_levels()}), not {level}")

    def teacher_grid(self, level: int) -> list:
        """The teacher's schedule from pure noise down to the student's level `level`."""
        self.check_level(level)
        grid = self.fine_grid()
        return grid[: grid.index(self.student_levels()[level]) + 1]

    def schedule(self, level: int | None) -> list:
        """The sampler: the adapted first jump to the student's level `level`, then the student's own remaining
        levels; level None is the released student's schedule. One denoiser evaluation per entry."""
        levels = self.student_levels()
        if level is None:
            return levels
        self.check_level(level)
        return levels[:1] + levels[level:]

    def sigmas(self, level: int) -> tuple[float, float]:
        """(noise at the start, noise at the student's level `level`) in first_step()'s latent coordinates."""
        return 1.0, float(self.student_levels()[level])

    def describe_level(self, level: int) -> str:
        return f"noise level {self.student_levels()[level]:.4f}"

    # --- loading -------------------------------------------------------------------------------------------------
    def load_student(self):
        """The student's pipeline on the CPU, its scheduler set to run explicit levels as given."""
        raise NotImplementedError

    def load_teacher(self):
        raise NotImplementedError

    def denoiser(self, pipe) -> nn.Module:
        return getattr(pipe, self.denoiser_name)

    def lora_layers(self, denoiser: nn.Module) -> list[str]:
        """Names of the linear layers that get a low-rank update."""
        raise NotImplementedError

    # --- noise, targets, prompts ---------------------------------------------------------------------------------
    def noise(self, seeds: list[int], device) -> torch.Tensor:
        """The starting noise of each image, drawn as the pipeline draws it (one generator per image on `device`), so
        image k keeps its noise whatever batch it is in."""
        raise NotImplementedError

    def teacher_states(self, pipe, prompts: list[str], seeds: list[int], level: int, guidance: float):
        """-> (x1, target) on the CPU: the starting latent and the teacher's latent at the student's level `level`.
        guidance is the standard w in neg + w (pos - neg); 1 is unguided (the paper's targets)."""
        raise NotImplementedError

    def text_encoders(self, pipe) -> list[nn.Module]:
        return [pipe.text_encoder]

    def encode(self, pipe, prompts: list[str], device) -> list[torch.Tensor]:
        """One embedding per prompt, as the pipeline computes it."""
        raise NotImplementedError

    def collate(self, embeddings: list[torch.Tensor], device):
        return torch.stack(embeddings).to(device)

    # --- training and generation ---------------------------------------------------------------------------------
    def training_denoiser(self, pipe, device) -> nn.Module:
        """The denoiser on `device`, frozen, ready for low-rank training."""
        net = self.denoiser(pipe).to(device)
        net.requires_grad_(False)
        net.enable_gradient_checkpointing()
        return net

    def first_step(self, pipe, net: nn.Module, level: int) -> Callable:
        """-> step(x1, embeddings, inference=False): the latent the denoiser's first prediction reaches at the student's
        level `level`, in float32. inference=True runs it in the precision of generation."""
        raise NotImplementedError

    def generate(self, pipe, prompts: list[str], seeds: list[int], schedule: list, callback=None):
        """PIL images of `prompts` from the seeds' noise on `schedule`; callback runs after every denoising step."""
        raise NotImplementedError


@functools.cache
def klein_levels(steps: int) -> tuple[float, ...]:
    """FLUX.2-klein's own noise levels for `steps` steps at 512 px (base and student share the rule): linear levels
    under the pipeline's resolution- and step-dependent exponential shift."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    e = np.exp(compute_empirical_mu((RESOLUTION // 16) ** 2, steps))
    s = np.linspace(1.0, 1 / steps, steps)
    return tuple(float(v) for v in e / (e + (1 / s - 1)))


@functools.cache
def zimage_levels(repo: str, steps: int) -> tuple[float, ...]:
    """Z-Image's own noise levels (base or Turbo): the pipeline's default levels under the scheduler's static shift
    (6 for the base, 3 for Turbo), as the scheduler's float32 values."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import get_default_z_image_sigmas
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
    if sch.config.use_dynamic_shifting:
        raise ValueError(f"{repo}: a resolution-dependent shift is not supported")
    sch.set_timesteps(sigmas=get_default_z_image_sigmas(steps))
    return tuple(sch.sigmas[:-1].tolist())


def generators(seeds, device) -> list[torch.Generator]:
    return [torch.Generator(torch.device(device)).manual_seed(int(s)) for s in seeds]


def explicit_levels(scheduler):
    """A flow-matching scheduler with its shift switched off, so explicit noise levels run as given."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_config(scheduler.config, use_dynamic_shifting=False, shift=1.0)


class StopAt:
    """callback_on_step_end that keeps the latents after step `step` and stops the denoising loop there."""

    def __init__(self, step: int):
        self.step, self.latents = step, None

    def __call__(self, pipe, i, t, kwargs):
        if i == self.step:
            self.latents = kwargs["latents"].detach().cpu()
            pipe._interrupt = True
        return kwargs


def device_of(pipe) -> torch.device:
    return pipe._execution_device


class Flux2Klein(Model):
    """FLUX.2-klein 4B, distilled to four steps, and its base."""

    name = "flux2-klein"
    student = "black-forest-labs/FLUX.2-klein-4B"
    teacher = "black-forest-labs/FLUX.2-klein-base-4B"
    student_steps = 4
    default_target_level = 3        # noise 0.7175: two denoiser evaluations
    dtype = torch.bfloat16
    batch_size = 8
    train_batch_size = 8
    # The base's 50-step levels with the student's levels 0.884 and 0.7175 in place of the base level each nearly
    # coincides with: 1 -> 0.7175 in 34 steps at 512 px, the grid of the paper's adapter. The student's level 0.958 is
    # not on it, so the jump can end at level 2 or 3.
    FINE = (("base", 0, 21), ("student", 2, 3), ("base", 22, 34), ("student", 3, 4))

    def student_levels(self):
        return list(klein_levels(self.student_steps))

    def fine_grid(self):
        own = {"base": klein_levels(50), "student": klein_levels(self.student_steps)}
        return [v for source, a, b in self.FINE for v in own[source][a:b]]

    def _load(self, repo):
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        pipe.scheduler = explicit_levels(pipe.scheduler)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def load_student(self):
        return self._load(self.student)

    def load_teacher(self):
        pipe = self._load(self.teacher)
        if pipe.config.is_distilled:
            raise ValueError(f"{self.teacher} is a distilled model; the teacher must be the base")
        return pipe

    def lora_layers(self, net):   # every linear layer of the double- and single-stream blocks
        return [n for n, m in net.named_modules() if isinstance(m, nn.Linear) and "transformer_blocks." in n]

    def noise(self, seeds, device):
        from diffusers.utils.torch_utils import randn_tensor
        channels = 128   # the transformer's packed channels: 4 x the VAE's 32
        return randn_tensor((len(seeds), channels, RESOLUTION // 16, RESOLUTION // 16), generator=generators(seeds, device),
                            device=torch.device(device), dtype=torch.bfloat16)

    def teacher_states(self, pipe, prompts, seeds, level, guidance):
        noise = self.noise(seeds, device_of(pipe))
        grid = self.teacher_grid(level)
        stop = StopAt(len(grid) - 2)
        pipe(prompt=list(prompts), latents=noise, sigmas=grid, num_inference_steps=len(grid), guidance_scale=guidance,
             height=RESOLUTION, width=RESOLUTION, output_type="latent", callback_on_step_end=stop)
        assert pipe.do_classifier_free_guidance == (guidance > 1)
        return pipe._pack_latents(noise).cpu(), stop.latents

    def encode(self, pipe, prompts, device):
        with torch.no_grad():
            return list(pipe.encode_prompt(list(prompts), device=torch.device(device))[0].unbind(0))

    def first_step(self, pipe, net, level):
        s0, s1 = self.sigmas(level)
        device = next(net.parameters()).device
        img_ids = pipe._prepare_latent_ids(torch.zeros(1, 128, RESOLUTION // 16, RESOLUTION // 16)).to(device)
        txt_ids = {}

        def step(x1, emb, inference=False):
            n = x1.shape[0]
            if emb.shape[1] not in txt_ids:
                txt_ids[emb.shape[1]] = pipe._prepare_text_ids(emb[:1]).to(device)
            v = net(hidden_states=x1, timestep=torch.ones(n, device=device, dtype=x1.dtype), guidance=None,
                    encoder_hidden_states=emb, txt_ids=txt_ids[emb.shape[1]].expand(n, -1, -1),
                    img_ids=img_ids.expand(n, -1, -1), return_dict=False)[0]
            return x1.float() + (s1 - s0) * v.float()
        return step

    def generate(self, pipe, prompts, seeds, schedule, callback=None):
        return pipe(prompt=list(prompts), latents=self.noise(seeds, device_of(pipe)), sigmas=schedule,
                    num_inference_steps=len(schedule), guidance_scale=1.0, height=RESOLUTION, width=RESOLUTION,
                    callback_on_step_end=callback).images


class SDTurbo(Model):
    """SD-Turbo, distilled from SD 2.1 with the same text encoder and VAE. Its levels are scheduler timesteps; the
    latents are x0 + s eps with s the Euler scheduler's noise scale (14.6 at timestep 999)."""

    name = "sd-turbo"
    student = "stabilityai/sd-turbo"
    teacher = "sd2-community/stable-diffusion-2-1-base"   # SD 2.1's UNet runs on SD-Turbo's pipeline
    student_steps = 4
    default_target_level = 2        # timestep 499: three denoiser evaluations
    dtype = torch.float16
    batch_size = 32
    train_batch_size = 16
    denoiser_name = "unet"
    TIMESTEPS = [999, 749, 499, 249]   # SD-Turbo's four-step schedule

    def student_levels(self):
        return list(self.TIMESTEPS)

    def fine_grid(self):
        """30 equal intervals from timestep 999 to 499, then 15 to 249, rounded to integers."""
        return ([int(round(t)) for t in np.linspace(999, 499, 31)] +
                [int(round(t)) for t in np.linspace(499, 249, 16)[1:]])

    def sigmas(self, level):
        from diffusers import EulerDiscreteScheduler
        sch = EulerDiscreteScheduler.from_pretrained(self.student, subfolder="scheduler")
        sch.set_timesteps(timesteps=[self.TIMESTEPS[0], self.TIMESTEPS[level]])
        return float(sch.sigmas[0]), float(sch.sigmas[1])

    def describe_level(self, level):
        return f"timestep {self.student_levels()[level]}"

    def load_student(self):
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(self.student, variant="fp16", torch_dtype=torch.float16)
        if type(pipe.scheduler).__name__ != "EulerDiscreteScheduler":
            raise ValueError(f"{self.student}: expected SD-Turbo's Euler scheduler, found {type(pipe.scheduler).__name__}")
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def load_teacher(self):
        from diffusers import UNet2DConditionModel
        pipe = self.load_student()
        pipe.unet = UNet2DConditionModel.from_pretrained(self.teacher, subfolder="unet", variant="fp16",
                                                         torch_dtype=torch.float16)
        return pipe

    def lora_layers(self, net):   # every linear layer of the UNet's attention blocks
        return [n for n, m in net.named_modules() if isinstance(m, nn.Linear) and "attentions." in n]

    def noise(self, seeds, device):
        from diffusers.utils.torch_utils import randn_tensor
        return randn_tensor((len(seeds), 4, RESOLUTION // 8, RESOLUTION // 8), generator=generators(seeds, device),
                            device=torch.device(device), dtype=torch.float16)

    def teacher_states(self, pipe, prompts, seeds, level, guidance):
        noise = self.noise(seeds, device_of(pipe))
        grid = self.teacher_grid(level)
        stop = StopAt(len(grid) - 2)
        pipe(prompt=list(prompts), latents=noise, timesteps=grid, guidance_scale=guidance if guidance > 1 else 0.0,
             height=RESOLUTION, width=RESOLUTION, output_type="latent", callback_on_step_end=stop)
        assert pipe.do_classifier_free_guidance == (guidance > 1)
        return (noise * pipe.scheduler.init_noise_sigma).cpu(), stop.latents   # the start the pipeline used

    def encode(self, pipe, prompts, device):
        with torch.no_grad():
            return list(pipe.encode_prompt(list(prompts), torch.device(device), 1, False)[0].unbind(0))

    def training_denoiser(self, pipe, device):
        net = self.denoiser(pipe).float().to(device)   # float32 weights, a bfloat16 forward
        net.requires_grad_(False)
        return net

    def first_step(self, pipe, net, level):
        s0, s1 = self.sigmas(level)
        device = next(net.parameters()).device
        t = torch.tensor(float(self.TIMESTEPS[0]), device=device)

        def step(x1, emb, inference=False):
            with torch.autocast(device.type, dtype=torch.float16 if inference else torch.bfloat16):
                eps = net(x1.float() / (s0 ** 2 + 1) ** 0.5, t, encoder_hidden_states=emb).sample
            return x1.float() + (s1 - s0) * eps.float()   # one Euler step in s, as SD-Turbo's scheduler takes it
        return step

    def generate(self, pipe, prompts, seeds, schedule, callback=None):
        return pipe(prompt=list(prompts), latents=self.noise(seeds, device_of(pipe)), timesteps=schedule,
                    guidance_scale=0.0, height=RESOLUTION, width=RESOLUTION, callback_on_step_end=callback).images


class ZImageTurbo(Model):
    """Z-Image-Turbo, distilled to nine steps, and its base Z-Image."""

    name = "z-image-turbo"
    student = "Tongyi-MAI/Z-Image-Turbo"
    teacher = "Tongyi-MAI/Z-Image"
    student_steps = 9
    default_target_level = 3        # noise 0.857: seven denoiser evaluations
    dtype = torch.bfloat16
    batch_size = 8
    train_batch_size = 8
    # The base's 50-step levels with the student's levels 0.96, 0.913 and 0.857 put in (the first and last in place of
    # the base level they coincide with): 1 -> 0.857 in 26 steps
    FINE = (("base", 0, 10), ("student", 1, 2), ("base", 11, 19), ("student", 2, 3), ("base", 19, 25), ("student", 3, 4))

    def student_levels(self):
        return list(zimage_levels(self.student, self.student_steps))

    def fine_grid(self):
        own = {"base": zimage_levels(self.teacher, 50), "student": zimage_levels(self.student, self.student_steps)}
        return [v for source, a, b in self.FINE for v in own[source][a:b]]

    def _load(self, repo):
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        pipe.scheduler = explicit_levels(pipe.scheduler)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def load_student(self):
        return self._load(self.student)

    def load_teacher(self):
        return self._load(self.teacher)

    def lora_layers(self, net):   # every linear layer of the main blocks and of the image and text refiners
        return [n for n, m in net.named_modules()
                if isinstance(m, nn.Linear) and n.split(".")[0] in ("layers", "noise_refiner", "context_refiner")]

    def noise(self, seeds, device):
        from diffusers.utils.torch_utils import randn_tensor
        return randn_tensor((len(seeds), 16, RESOLUTION // 8, RESOLUTION // 8), generator=generators(seeds, device),
                            device=torch.device(device), dtype=torch.float32)

    def teacher_states(self, pipe, prompts, seeds, level, guidance):
        noise = self.noise(seeds, device_of(pipe))
        grid = self.teacher_grid(level)
        stop = StopAt(len(grid) - 2)
        # Z-Image's pipeline computes pos + g (pos - neg): its g is w - 1. Prompt lists go in as copies because the
        # pipeline writes its chat template into the list it is given.
        pipe(prompt=list(prompts), latents=noise, sigmas=grid, num_inference_steps=len(grid), guidance_scale=guidance - 1.0,
             height=RESOLUTION, width=RESOLUTION, output_type="latent", callback_on_step_end=stop)
        assert pipe.do_classifier_free_guidance == (guidance > 1)
        return noise.cpu(), stop.latents

    def encode(self, pipe, prompts, device):   # one embedding per prompt, of its own length
        with torch.no_grad():
            return list(pipe.encode_prompt(list(prompts), torch.device(device), False)[0])

    def collate(self, embeddings, device):
        return [e.to(device) for e in embeddings]

    def first_step(self, pipe, net, level):
        s0, s1 = self.sigmas(level)
        device = next(net.parameters()).device

        def step(x1, emb, inference=False):
            # the transformer takes a list of (C, 1, H, W) latents and the timestep (1000 - t) / 1000, 0 at pure noise,
            # and returns minus the velocity
            n = x1.shape[0]
            out = net(list(x1.to(torch.bfloat16).unsqueeze(2).unbind(0)), torch.zeros(n, device=device), emb,
                      return_dict=False)[0]
            return x1.float() - (s1 - s0) * torch.stack(out).float().squeeze(2)
        return step

    def generate(self, pipe, prompts, seeds, schedule, callback=None):
        return pipe(prompt=list(prompts), latents=self.noise(seeds, device_of(pipe)), sigmas=schedule,
                    num_inference_steps=len(schedule), guidance_scale=0.0, height=RESOLUTION, width=RESOLUTION,
                    callback_on_step_end=callback).images


MODELS = {m.name: m for m in (Flux2Klein, SDTurbo, ZImageTurbo)}


def get_model(name: str, student: str | None = None, teacher: str | None = None) -> Model:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](student, teacher)
