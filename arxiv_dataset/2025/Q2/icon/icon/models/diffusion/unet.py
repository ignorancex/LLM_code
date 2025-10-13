"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
Copyright (c) 2023 Stony Brook Robotics Lab
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from typing import List, Dict, Union
from einops import rearrange
from icon.models.diffusion.positional_embedding import SinusoidalPosEmb
from icon.models.diffusion.conv_components import (
    Downsample1d,
    Upsample1d,
    Upsample2d,
    Conv1dBlock,
    Conv2dBlock,
    ConditionalResidualBlock1D,
    ResidualBlock2D
)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py#L69
class ConditionalUnet1D(nn.Module):

    def __init__(
        self, 
        input_dim: int,
        obs_cond_dim: int,
        timestep_embed_dim: int,
        down_dims: List,
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.timestep_embed = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        cond_dim = timestep_embed_dim + obs_cond_dim
        all_dims = [input_dim] + down_dims
        in_out_dims = list(zip(all_dims[:-1], all_dims[1:]))
        common_kwargs = dict(
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups
        )

        self.down_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(in_out_dims):
            is_last = idx >= (len(in_out_dims) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(in_dim, out_dim, **common_kwargs),
                    ConditionalResidualBlock1D(out_dim, out_dim, **common_kwargs),
                    Downsample1d(out_dim) if not is_last else nn.Identity()
                ])
            )
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], **common_kwargs),
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], **common_kwargs)
        ])
        self.up_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(reversed(in_out_dims[1:])):
            is_last = idx >= (len(in_out_dims) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(out_dim * 2, in_dim, **common_kwargs),
                    ConditionalResidualBlock1D(in_dim, in_dim, **common_kwargs),
                    Upsample1d(in_dim) if not is_last else nn.Identity()
                ])
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        obs_cond: Tensor
    ) -> Tensor:
        """
        Args:
            timesteps (torch.Tensor): diffusion timesteps (batch_size,).
            obs_cond (torch.Tensor): observation conditionings (batch_size, obs_cond_dim).
        """
        timestep_embed = self.timestep_embed(timesteps)
        cond = torch.cat([timestep_embed, obs_cond], dim=1)
        x = x.permute(0, 2, 1)
        residuals = list()
        for block1, block2, downsample in self.down_modules:
            x = block1(x, cond)
            x = block2(x, cond)
            residuals.append(x)
            x = downsample(x)
        for block in self.mid_modules:
            x = block(x, cond)
        for block1, block2, upsample in self.up_modules:
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x
    
# Adapted from https://github.com/LostXine/crossway_diffusion/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py#L282
class ConditionalUnet1DwDec(ConditionalUnet1D):

    def __init__(
        self,
        input_dim: int,
        obs_cond_dim: int,
        timestep_embed_dim: int,
        down_dims: List,
        kernel_size: int,
        n_groups: int,
        obs_horizon: int,
        cameras: List,
        shape_meta: Dict,
        decode_dims: List,
        decode_low_dim_dims: List,
        decode_pe_dim: int,
        decode_resolution: int
    ) -> None:
        super().__init__(
            input_dim,
            obs_cond_dim,
            timestep_embed_dim,
            down_dims,
            kernel_size,
            n_groups
        )
        # Image decoder
        self.image_size = shape_meta['images']
        all_dims = [input_dim] + list(decode_dims)
        in_out_dims = list(zip(all_dims[:-1], all_dims[1:]))
        self.image_decoders = nn.ModuleDict({})
        self.final_convs = nn.ModuleDict({})
        for camera in cameras:
            image_decoder = nn.ModuleList([])
            for idx, (in_dim, out_dim) in enumerate(reversed(in_out_dims[1:])):
                is_last = idx >= (len(in_out_dims) - 1)
                image_decoder.append(
                    nn.Sequential(
                        ResidualBlock2D(out_dim + decode_pe_dim, in_dim, kernel_size, n_groups),
                        ResidualBlock2D(in_dim, in_dim, kernel_size, n_groups),
                        Upsample2d(in_dim) if not is_last else nn.Identity()
                    )
                )
            final_conv = nn.Sequential(
                Conv2dBlock(decode_dims[0], decode_dims[0], kernel_size, n_groups),
                nn.Conv2d(decode_dims[0], 3, 1),
            ) 
            self.image_decoders[camera] = image_decoder
            self.final_convs[camera] = final_conv
        # Low-dim decoder
        low_dim_channel = shape_meta['low_dims']
        in_out_dims = list(zip(decode_low_dim_dims[:-1], decode_low_dim_dims[1:]))
        self.low_dim_decoder = list()
        self.low_dim_decoder.append(nn.Linear(down_dims[-1] // obs_horizon, low_dim_channel * decode_low_dim_dims[0]))
        for in_dim, out_dim in in_out_dims:
            self.low_dim_decoder.append(nn.Mish())
            self.low_dim_decoder.append(nn.Linear(low_dim_channel * in_dim, low_dim_channel * out_dim))
        self.low_dim_decoder = nn.Sequential(*self.low_dim_decoder)

        self.obs_horizon = obs_horizon
        self.cameras = cameras
        self.decode_resolution = decode_resolution
        self.decode_pe_dim = decode_pe_dim

    def generate_positional_embedding(self, x: Tensor, dim: int) -> Tensor:
        b, _, h, w = x.shape
        hidx = torch.linspace(-1, 1, steps=h)
        widx = torch.linspace(-1, 1, steps=w)
        freq = dim // 4
        sh = [(2 ** i) * torch.pi * hidx for i in range(freq)]
        sw = [(2 ** i) * torch.pi * widx for i in range(freq)]
        grids = [torch.stack(torch.meshgrid(hi, wi, indexing='ij'), axis=0) for hi, wi in zip(sh, sw)]
        phases = torch.concat(grids, 0)
        assert phases.shape == (dim // 2, h, w)
        pe = torch.concat([torch.sin(phases), torch.cos(phases)], axis=0)
        bpe = pe.unsqueeze(0).repeat(b, 1, 1, 1)
        bpe = bpe.to(x.device)
        return bpe

    def forward_mid_features(
        self,
        x: Tensor,
        timesteps: Tensor,
        obs_cond: Tensor
    ) -> Tensor:
        timestep_embed = self.timestep_embed(timesteps)
        cond = torch.cat([timestep_embed, obs_cond], dim=1)
        x = x.permute(0, 2, 1)
        residuals = list()
        for block1, block2, downsample in self.down_modules:
            x = block1(x, cond)
            x = block2(x, cond)
            residuals.append(x)
            x = downsample(x)
        for block in self.mid_modules:
            x = block(x, cond)
        mid = x.clone()
        for block1, block2, upsample in self.up_modules:
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x, mid
    
    def forward_decoder(self, mid: Tensor) -> Dict:
        mid_pre = mid[:, :, 0]
        mid_ld = rearrange(mid_pre, 'b (t c) -> (b t) c', t=self.obs_horizon)
        mid_rgb = mid_ld.reshape(mid_ld.shape[0], -1, self.decode_resolution, self.decode_resolution)
        recons = dict()
        # Image reconstruction
        image_recons = dict()
        for camera in self.cameras:
            image_decoder = self.image_decoders[camera]
            n_upsamples = len(image_decoder)
            h_res = w_res = self.image_size // (2 ** n_upsamples)
            h_scale = w_scale = math.ceil(h_res / self.decode_resolution)
            x = mid_rgb.repeat(1, 1, h_scale, w_scale)
            x = x[:, :, :h_res, :w_res]
            for block in image_decoder:
                pos_embed = self.generate_positional_embedding(x, self.decode_pe_dim)
                x = torch.cat([x, pos_embed], dim=1)
                x = block(x)
            x = self.final_convs[camera](x)
            image_recons[camera] = rearrange(x, '(b t) ... -> b t ...', t=self.obs_horizon)
        recons['images'] = image_recons
        # Low-dim reconstruction
        recons['low_dims'] = rearrange(self.low_dim_decoder(mid_ld), '(b t) ... -> b t ...', t=self.obs_horizon)
        return recons
    
    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        obs_cond: Tensor,
        obs_targets: Union[Dict, None] = None
    ) -> Tensor:
        x, mid = self.forward_mid_features(x, timesteps, obs_cond)
        if obs_targets is None:
            return x
        else:
            recons = self.forward_decoder(mid)
            recons_loss = 0
            for camera in self.cameras:
                image_recons = recons['images'][camera]
                image_target = obs_targets['images'][camera]
                image_recons_loss = F.mse_loss(image_recons, image_target)
                recons_loss += image_recons_loss
            recons_loss += F.mse_loss(recons['low_dims'], obs_targets['low_dims'])
            return x, recons_loss