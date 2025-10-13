import torch
from tqdm import tqdm

from v2sflow.registry import SCHEDULERS

from .rectified_flow import RFlowAudioScheduler_x0


@SCHEDULERS.register_module("rflow_audio_x0")
class RFLOW_audio_x0:
    def __init__(
        self,
        cfg_scale=4.0,
        **kwargs,
    ):
        self.cfg_scale = cfg_scale
        self.scheduler = RFlowAudioScheduler_x0(
            **kwargs,
        )

    def sample(
        self,
        model,
        z,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        model_args = additional_args
        model_args["cfg_mask"] = torch.cat([torch.ones(z.shape[0]), torch.zeros(z.shape[0])], 0).bool().to(z.device)

        # prepare timesteps
        timesteps = [(1.0 - i / self.scheduler.num_sampling_steps) * self.scheduler.num_timesteps for i in range(self.scheduler.num_sampling_steps)]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            if i == 0:
                noise = z.clone()
            else:
                timepoints = timesteps[i].float() / self.scheduler.num_timesteps
                noise = (z - (1 - timepoints) * (v_pred + noise)) / timepoints

            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.scheduler.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                # model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)

            x_pred = model(z_in, t, **model_args)#.chunk(2, dim=1)[0]
            pred = x_pred - noise

            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.scheduler.num_timesteps
            z = z + v_pred * dt[:, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :], z, x0)

        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, t)
