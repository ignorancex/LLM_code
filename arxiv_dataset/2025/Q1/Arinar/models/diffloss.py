import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion
from models.cond_mlp import SimpleMLPAdaLN


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, token_embed_dim, decoder_embed_dim, head_depth, head_width, 
                 num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = token_embed_dim
        self.net = SimpleMLPAdaLN(
            in_channels=token_embed_dim,
            model_channels=head_width,
            out_channels=token_embed_dim * 2,  # for vlb loss
            z_channels=decoder_embed_dim,
            num_res_blocks=head_depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift
