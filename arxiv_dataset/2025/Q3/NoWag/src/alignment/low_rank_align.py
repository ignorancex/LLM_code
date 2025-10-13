import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import argparse
from copy import deepcopy
import src.utils.alignment.grads as grads


@torch.enable_grad()
def align_low_rank(
    original_weight: torch.Tensor,
    reconstruct_fn: callable,
    reconstruct_params: dict[str, torch.Tensor],
    hessian: torch.Tensor,
    regularization_lambda: float,
    epochs: int,
    lr: float,
    lr_multiplier: float,
    parameters_to_exclude: list[str] = [],
    early_stop_eps: float = 1e-7,
    patience: int = 100,
    verbose: int = -1,
) -> dict[str, torch.Tensor]:
    """aligns the low rank to be close to the original weight on the callibration dataset

    Args:
        original_weight (torch.Tensor): _description_
        reconstruct_fn (nn.Module | callable): _description_
        reconstruct_params (dict[str, torch.Tensor]): _description_
        hessian (torch.Tensor): _description_
        args (argparse.Namespace): _description_
        parameters_to_exclude (list[str], optional): _description_. Defaults to [].
        verbose (int, optional): _description_. Defaults to -1.
    """

    params_to_optimize = {}
    for name, param in reconstruct_params.items():
        if isinstance(param, torch.Tensor):
            if param.requires_grad and name not in parameters_to_exclude:
                print(name)
                params_to_optimize[name] = param

    # name: param for name, param in reconstruct_params.items() if param.requires_grad and name not in parameters_to_exclude}
    # print("params to optimize", params_to_optimize)
    # print(lr)
    optimizer = torch.optim.Adam(params_to_optimize.values(), lr=lr)
    optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=lr_multiplier
    )

    hessain_to_use = hessian.clone() / hessian.shape[0]
    ids = torch.arange(hessian.shape[0], device=hessian.device)
    hessain_to_use[ids, ids] += regularization_lambda

    prev_loss = float("inf")
    remaining_patience = patience
    for epoch in range(epochs):
        optimizer.zero_grad()

        reconstructed_weights = reconstruct_fn(**reconstruct_params)
        # print(reconstructed_weights)
        diff = original_weight - reconstructed_weights
        # print(diff)
        # print(diff)
        loss = torch.einsum("ij,jk,ik->", diff, hessain_to_use, diff)
        # print(loss)
        loss.backward()
        optimizer.step()

        if loss.item() > prev_loss - early_stop_eps:
            remaining_patience -= 1
            if remaining_patience == 0:
                print("early stopping at epoch", epoch)
                break
            optimizer_scheduler.step()
        else:
            remaining_patience = patience
            prev_loss = loss.item()
            best_weights = deepcopy(reconstruct_params)

        if verbose > 0 and epoch % verbose == 0:
            print(f"Epoch {epoch} Loss {loss.item()}")

        if loss < 0:
            print("breaking because the loss is negative")
    # print(params_to_optimize)
    print("last loss", loss.item())
    # raise ValueError("done")
    return best_weights


def align_simple(
    original_weight,
    A,
    B,
    lr,
    hessian,
    regularization_lambda,
    epochs,
    lr_multiplier,
    early_stop_eps=1e-7,
    patience: int = 100,
    verbose: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer = torch.optim.Adam([A, B], lr=lr)

    optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=lr_multiplier
    )

    hessain_to_use = hessian.clone()

    ids = torch.arange(hessian.shape[0], device=hessian.device)
    hessain_to_use[ids, ids] += regularization_lambda

    prev_loss = float("inf")
    remaining_patience = patience
    for epoch in range(epochs):
        optimizer.zero_grad()

        reconstructed_weights = A @ B

        diff = original_weight - reconstructed_weights
        loss = torch.einsum("ij,jk,ik->", diff, hessain_to_use, diff)

        # gradients use the hardcoded gradients
        A_grad, B_grad = grads.grad_quadratic_low_rank(
            A, B, original_weight, hessain_to_use
        )

        A.grad = A_grad
        B.grad = B_grad

        optimizer.step()

        if loss.item() > prev_loss - early_stop_eps:
            remaining_patience -= 1
            if remaining_patience == 0:
                print("early stopping at epoch", epoch)
                break
            optimizer_scheduler.step()

        elif loss < early_stop_eps:
            print("breaking because the loss is negative")
            break

        else:
            remaining_patience = patience
            prev_loss = loss.item()
            best_A = A.clone()
            best_B = B.clone()

        # if verbose > 0 and epoch % verbose == 0:
        #     print(f"Epoch {epoch} Loss {loss.item()}")

    print("last loss", loss.item(), "best loss", prev_loss)
    return best_A, best_B
