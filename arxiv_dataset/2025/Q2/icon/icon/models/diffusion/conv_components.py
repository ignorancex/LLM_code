"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
from torch import nn
from torch import Tensor
from einops import rearrange


# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py#L7
class Downsample1d(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py#L15
class Upsample1d(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    
# Adapted from https://github.com/LostXine/crossway_diffusion/blob/main/diffusion_policy/model/diffusion/conv2d_components.py#L6
class Upsample2d(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py#L23
class Conv1dBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
    
# Adapted from https://github.com/LostXine/crossway_diffusion/blob/main/diffusion_policy/model/diffusion/conv2d_components.py#L14
class Conv2dBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

# Adapted from https://github.com/LostXine/crossway_diffusion/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py#L254
class ResidualBlock2D(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residuals = self.residual_conv(x)
        x = self.blocks(x) + residuals
        return x

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py#L14
class ConditionalResidualBlock1D(nn.Module):

    def __init__(
        self,  
        in_channels: int, 
        out_channels: int, 
        cond_dim: int,
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        residuals = self.residual_conv(x)
        cond = self.cond_encoder(cond)
        cond = rearrange(cond, 'b (d t) -> b d t', t=2)
        scale, bias = cond.chunk(2, dim=-1)
        x = self.blocks[0](x)
        x = x * scale + bias
        x = self.blocks[1](x)
        x = x + residuals
        return x