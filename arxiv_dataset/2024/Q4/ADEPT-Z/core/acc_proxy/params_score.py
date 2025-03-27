"""
Date: 2023-10-19 11:12:15
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:33:57
FilePath: /ADEPT_Zero/core/acc_proxy/params_score.py
"""
import torch
from torch import nn

__all__ = ["compute_params_score"]


def compute_params_score(
    model: nn.Module,
    resolution: int = 32,
    batch_size: int = 1,
    device="cuda:0",
    # device = "cpu",
    fp16: bool = False,
    in_channels: int = 3,
):  # forward/backward
    model.train()
    model.requires_grad_(True)
    model.to(device)
    model.zero_grad()

    if isinstance(device, str):
        device = torch.device(device)
    with torch.autocast(device.type, enabled=fp16):
        # network_weight_gaussian_init(model)
        input = torch.randn(
            size=[1, in_channels, resolution, resolution], device=device
        )
        output = model(input)
    output.sum().backward()
    Total_params = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, "grad") and p.grad is not None:
                Total_params = Total_params + p.grad.numel()
    return Total_params / sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
