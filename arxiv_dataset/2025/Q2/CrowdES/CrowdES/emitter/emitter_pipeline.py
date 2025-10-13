import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from CrowdES.emitter.emitter_model import CrowdESEmitterModel
from utils.image import sampling_xy_pos


logger = logging.get_logger(__name__)


@dataclass
class CrowdESEmitterPipelineOutput(BaseOutput):
    crowd_emission: Union[torch.Tensor, np.ndarray]


class CrowdESEmitterPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        unet: CrowdESEmitterModel,
        scheduler: Union[
            DDPMScheduler,
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        condition: torch.Tensor,
        num_crowd: int = 10,
        num_inference_steps: int = 50,
        batch_size: int = 1,
        guidance_scale: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = 'tensor',
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        eta: float = 1.0,
        leapfrog_steps: int = 5,
        post_process: bool = True,

        **kwargs,
    ):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare noise input
        max_num_agents = self.unet.config.max_num_agents
        num_classes =  self.unet.config.num_classes
        condition_size =  self.unet.config.condition_size
        condition_channels =  self.unet.config.condition_channels
        vae_latent_dim = self.unet.config.vae_latent_dim
        latent_len_multiplier = self.unet.config.latent_len_multiplier
        condition = torch.cat([condition] * batch_size)
        
        # Reproducibility
        generator = torch.Generator().manual_seed(seed)
        pred_crowd = torch.randn((batch_size, max_num_agents, num_classes * latent_len_multiplier), generator=generator).to(device)
        pred_crowd_mask = torch.zeros((batch_size, max_num_agents), dtype=torch.bool, device=device)
        pred_crowd_mask[:, :num_crowd] = 1
        pred_crowd.masked_fill_(~pred_crowd_mask.unsqueeze(-1), 0)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Leapfrog
        if leapfrog_steps is not None and leapfrog_steps < num_inference_steps:
            timesteps = timesteps[-leapfrog_steps:]
            appearance_density = condition[[0], -2]
            # frame_origin, agent_type, agent_velocity, origin_x, origin_y, goal_x, goal_y
            pred_crowd_lf = torch.zeros((batch_size, max_num_agents, num_classes), device=device)
            pred_crowd_lf[:, :, 1] = torch.rand((batch_size, max_num_agents), device=device)  # frame_origin
            retry = 100
            xy_samples = sampling_xy_pos(appearance_density, batch_size * num_crowd * retry, rel_threshold=0.1, replacement=True)
            xy_samples[..., 0] /= appearance_density.shape[2]
            xy_samples[..., 1] /= appearance_density.shape[1]
            xy_samples = xy_samples.view(batch_size, num_crowd, retry, 2)
            origin_xy = xy_samples[:, :, 0]
            goal_xy = xy_samples[:, :, 1]
            xy_samples_left = xy_samples[:, :, 2:]

            # If origin and goal are too close, replace goal with the next sample
            while xy_samples_left.size(-2) > 0:
                origin_goal_diff = torch.norm(origin_xy - goal_xy, dim=-1)
                mask = origin_goal_diff < 0.5

                if mask.sum() == 0:
                    break
                
                goal_xy[mask, :] = xy_samples_left[:, :, 0][mask, :]
                xy_samples_left = xy_samples_left[:, :, 1:]

            pred_crowd_lf[:, :num_crowd, [3, 4]] = origin_xy
            pred_crowd_lf[:, :num_crowd, [5, 6]] = goal_xy
            pred_crowd_lf = self.unet.VAEEncoder(pred_crowd_lf)

            pred_crowd_lf[:, :, [i*num_classes+0 for i in range(latent_len_multiplier)]] = torch.randn((batch_size, max_num_agents, 1), device=device)  # agent_type
            pred_crowd_lf[:, :, [i*num_classes+2 for i in range(latent_len_multiplier)]] = torch.randn((batch_size, max_num_agents, 1), device=device)  # agent_velocity
            pred_crowd_lf.masked_fill_(~pred_crowd_mask.unsqueeze(-1), 0)
            pred_crowd = pred_crowd_lf

            # Prepare the scaling factor
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device).to(dtype=pred_crowd.dtype)
            sqrt_alpha_prod = (alphas_cumprod[timesteps] ** 0.5).tolist()
            sqrt_one_minus_alpha_prod = ((1 - alphas_cumprod[timesteps]) ** 0.5)
            beta = [r / l * (1 + 1 / (num_inference_steps // leapfrog_steps)) for l, r in zip(sqrt_alpha_prod, sqrt_alpha_prod[1:] + [1])]
            scale = torch.tensor(beta).prod() * (1 - sqrt_one_minus_alpha_prod).prod()

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        # expand the condition if we are doing classifier free guidance
        condition_model_input = torch.cat([torch.zeros_like(condition), condition]) if do_classifier_free_guidance else condition
        pred_crowd_mask_model_input = torch.cat([pred_crowd_mask] * 2) if do_classifier_free_guidance else pred_crowd_mask
            
        for i, t in enumerate(timesteps):
            t_model_input = torch.stack([t] * batch_size * 2) if do_classifier_free_guidance else torch.stack([t] * batch_size)
            pred_crowd_model_input = torch.cat([pred_crowd] * 2) if do_classifier_free_guidance else pred_crowd

            # Scale the input
            pred_crowd_model_input = self.scheduler.scale_model_input(pred_crowd_model_input, t)
            
            # Predict the noise residual
            noise_pred = self.unet(
                pred_crowd_model_input, 
                mask = pred_crowd_mask_model_input,
                condition=condition_model_input,
                timestep=t_model_input,
            ).sample.to(dtype=torch.float32)
            noise_pred.masked_fill_(~pred_crowd_mask_model_input.unsqueeze(-1), 0)

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            pred_crowd = self.scheduler.step(noise_pred, t, pred_crowd, **extra_step_kwargs).prev_sample
            # pred_crowd = noise_pred  # x0 mode

            # Call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, pred_crowd)
        
        # Decode for latent diffusion model
        pred_crowd[:, :, 1::num_classes] *= 1.7
        pred_crowd = self.unet.VAEDecoder(pred_crowd)
        pred_crowd[:, :num_crowd, [3, 4]] = origin_xy
        pred_crowd[:, :num_crowd, [5, 6]] = goal_xy

        # Post-processing
        if post_process:
            # Truncate the output with num_crowd
            pred_crowd = pred_crowd[:, :num_crowd]
            pred_crowd[:, :, 0] = torch.clamp(pred_crowd[:, :, 0], 0, 2)             # round agent_type to 0, 1 or 2
            pred_crowd[:, :, 0] = torch.round(pred_crowd[:, :, 0])                   # Make it to descrete
            pred_crowd[:, :, 1] = torch.clamp(pred_crowd[:, :, 1], 0, 1)             # truncate frame_origin to 0~1
            pred_crowd[:, :, 2] = torch.clamp(pred_crowd[:, :, 2], 0, float('inf'))  # truncate preferred_speed to 0~inf
            pred_crowd[:, :, 3:] = torch.clamp(pred_crowd[:, :, 3:], 0, 1)           # truncate xy pos to 0~1

            # Sort with frame_origin, consider batch, max_num_agents
            frame_origin = pred_crowd[:, :, 1]
            sorted_indices = torch.argsort(frame_origin, dim=1, descending=False)
            pred_crowd = torch.gather(pred_crowd, 1, sorted_indices.unsqueeze(-1).expand_as(pred_crowd))

        if not return_dict:
            return pred_crowd

        return CrowdESEmitterPipelineOutput(crowd_emission=pred_crowd)
