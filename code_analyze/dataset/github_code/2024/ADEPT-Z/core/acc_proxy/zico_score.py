"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-08-15 16:03:46
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:35:04
"""

import os
import sys
from typing import Callable

from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import nn

__all__ = ["compute_zico_score"]


def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, SuperBlockConv2d) or isinstance(mod, SuperBlockLinear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                if torch.is_complex(mod.weight.grad):
                    weight_grad = torch.view_as_real(mod.weight.grad.data).view(-1)
                else:
                    weight_grad = mod.weight.grad.data.view(-1)
                grad = torch.cat(
                    [weight_grad]
                    + [
                        layer.weight.grad.data.view(-1)
                        for layer in mod.super_ps_layers
                        if layer.weight.grad is not None
                    ]
                )
                grad_dict[name] = [grad.cpu().numpy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, SuperBlockConv2d) or isinstance(mod, SuperBlockLinear):
                if torch.is_complex(mod.weight.grad):
                    weight_grad = torch.view_as_real(mod.weight.grad.data).view(-1)
                else:
                    weight_grad = mod.weight.grad.data.view(-1)
                grad = torch.cat(
                    [weight_grad]
                    + [
                        layer.weight.grad.data.view(-1)
                        for layer in mod.super_ps_layers
                        if layer.weight.grad is not None
                    ]
                )
                grad_dict[name].append(grad.cpu().numpy())
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array = None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(
                np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            )
    return nsr_mean_sum_abs


def update_grad_stat(grad_running_stat, model):
    all_grads = []  # collect gradient information of all modules(Conv or Linear) in the model

    for _, m in model.named_modules():
        if isinstance(m, SuperBlockConv2d) or isinstance(m, SuperBlockLinear):
            if torch.is_complex(m.weight.grad):
                weight_grad = torch.view_as_real(m.weight.grad.data).view(-1)
            else:
                weight_grad = m.weight.grad.data.view(-1)

            grad = torch.cat(
                [weight_grad]
                + [
                    layer.weight.grad.data.view(-1)
                    for layer in m.super_ps_layers
                    if layer.weight.grad is not None
                ]
            )

            all_grads.append(grad)

    # Concatenate all gradients into one large vector: gradient information for the model(average of batch size)
    total_grad = torch.cat(all_grads)

    # update grad_running_stat
    if "sum_grad" not in grad_running_stat:
        grad_running_stat["sum_grad"] = (
            total_grad  # average of gradient for all data in the batch
        )
        grad_running_stat["sum_grad_sq"] = (
            total_grad.square()
        )  # the square of average gradient for all data in the batch
        grad_running_stat["sum_grad_abs"] = (
            total_grad.abs()
        )  # the absolute of average gradient for all data in the batch
    else:
        grad_running_stat["sum_grad"] += total_grad
        grad_running_stat["sum_grad_sq"] += total_grad.square()
        grad_running_stat["sum_grad_abs"] += total_grad.abs()


def calculate_zico_score(
    grad_running_stat, num_batches
) -> float:  # get sum of gradients on the whole dataset
    nsr_mean_sum_abs = 0
    # average over all batches.
    grad_abs_mean = grad_running_stat["sum_grad_abs"] / num_batches
    grad_std = (
        grad_running_stat["sum_grad_sq"] / num_batches
        - (grad_running_stat["sum_grad"] / num_batches).square()
    ).sqrt()
    non_zeros = torch.nonzero(grad_std)[:, 0]
    ratio = (grad_abs_mean[non_zeros] / grad_std[non_zeros]).sum()

    if ratio > 0:
        nsr_mean_sum_abs += torch.log(ratio)
    return nsr_mean_sum_abs


def compute_zico_score(
    model: nn.Module,
    calib_dataloader,  # use validation dataset
    criterion: Callable,
    fp16: bool = False,
    device=torch.device("cuda:0"),
) -> float:
    # grad_dict = {}
    grad_running_stat = {}  # {"sum_grad": 0, "sum_grad_sq": 0, "sum_grad_abs": 0} # runing sum(grad), sum(grad.abs()), sum(grad.square())
    is_train = model.training
    model.train()
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)

    for i, batch in enumerate(calib_dataloader):
        model.zero_grad()
        with torch.autocast(device_type=device.type, enabled=fp16):
            data, target = batch[0], batch[1]
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(data)
            loss = criterion(logits, target)
        loss.backward()
        # grad_dict = getgrad(model, grad_dict, i)

        update_grad_stat(grad_running_stat, model)  # update grad stat

    # total_batch = len(calib_dataloader)
    # print("dataset_size:", dataset_size)
    nsr_mean_sum_abs = calculate_zico_score(
        grad_running_stat, len(calib_dataloader)
    )  # calculate zico-score for the whole dataset
    model.zero_grad()  # empty gradient

    model.train(is_train)
    return nsr_mean_sum_abs.item()
