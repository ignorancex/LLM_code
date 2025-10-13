import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse


@torch.enable_grad()
def align_cluster_one_step(
    X: torch.Tensor,
    centriods: torch.Tensor,
    reconstruction_fn: callable,
    reconstruction_additional_params: dict[str, torch.Tensor],
    hessian: torch.Tensor,
    centriods_optimizer: torch.optim.Optimizer,
    clip_grad: float = 0,
) -> tuple[float, torch.Tensor]:
    """aligns the centriods for one iteration on the callibration dataset

    Args:
        X (torch.Tensor): _description_
        centriods (torch.Tensor): _description_
        reconstruction_fn (nn.Module | callable): _description_
        reconstruction_additional_params (dict[str, torch.Tensor]): _description_
        hessian (torch.Tensor): _description_
        alignment_lr (float): _description_
        args (argparse.Namespace): _description_

    Returns:
        tuple[float, torch.Tensor]: _description_
    """

    reconstructed_weights = reconstruction_fn(
        centriods=centriods, **reconstruction_additional_params
    )
    # print(reconstructed_weights.dtype)
    diff = X - reconstructed_weights
    # print(diff.dtype)
    # print(hessian.dtype)
    loss = torch.einsum("ij,jk,ik->", diff, hessian, diff)
    loss.backward()

    # get the gradient of the centriods

    # if args clip the gradients
    if clip_grad > 0:
        centriods.grad = torch.clamp(centriods.grad, -clip_grad, clip_grad)
    centriods_optimizer.step()
    centriods_optimizer.zero_grad()
    # with torch.no_grad():
    #     #update the centriods
    #     centriods -= alignment_lr * centriods.grad
    #     centriods.grad.zero_()
    #     # print("centriods", centriods)
    # print(loss.item())
    return loss.item(), centriods
