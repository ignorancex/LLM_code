import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, MultiStepLR, SequentialLR
from transformers import get_inverse_sqrt_schedule


def inverse_sqrt_with_warmup_lr_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int = 10000,
    timescale: int = 1,
    last_epoch: int = -1,
) -> SequentialLR:
    """
    T5 learning rate scheduler with the original hyper parameter (not necessarily fits out use case)

    '''
    During pre-training, we use an “inverse square root” learning rate schedule: 1/sqrt(max(n, k))
    where n is the current training iteration and k is the number of warm-up steps (set to 10^4 in all of our experiments).
    This sets a constant learning rate of 0.01 for the first 104 steps, then exponentially decays the learning rate until pre-training is over.
    '''
    """
    constant_sch = ConstantLR(optimizer, factor=1.0, total_iters=num_warmup_steps)
    inverse_sqrt_schedule = get_inverse_sqrt_schedule(
        optimizer=optimizer,
        num_warmup_steps=0,
        timescale=timescale,
        last_epoch=last_epoch,
    )
    return SequentialLR(
        optimizer,
        schedulers=[constant_sch, inverse_sqrt_schedule],
        milestones=[num_warmup_steps],
    )


def cosine_annealing_with_warmup_lr_scheduler(
    optimizer: Optimizer,
    *,
    num_warmup_steps: int = 2000,
    start_factor: float = 0.3333333333333333,
    T_max: int = 40000,
    eta_min_factor: float = 0.1,
) -> SequentialLR:
    """
    cosine annealing with warmup, followed by a constant learning rate
    num_warmup_steps: warmup period, number of steps until the learning rate reach the maximum
    start_factor: the factor to start with in warmup period
    T_max: number of steps until the cosine reach the minimum
    eta_min_factor: minimum learning factor of the cosine scheduler
    """
    linear_sch = LinearLR(
        optimizer, start_factor=start_factor, total_iters=num_warmup_steps
    )
    assert (
        len(optimizer.param_groups) == 1
    ), f"this learning rate scheduler support single params group, got {optimizer.param_groups=}"

    initial_lr = [group["initial_lr"] for group in optimizer.param_groups][0]
    eta_min = eta_min_factor * initial_lr
    cosine_lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )
    multi_step_lr_sch = MultiStepLR(
        optimizer=optimizer, milestones=[0], gamma=eta_min_factor
    )
    return SequentialLR(
        optimizer,
        schedulers=[linear_sch, cosine_lr_sch, multi_step_lr_sch],
        milestones=[num_warmup_steps, T_max],
    )


def multistep_with_warmup_lr_scheduler(
    optimizer: Optimizer,
    *,
    num_warmup_steps: int = 2000,
    start_factor: float = 0.3333333333333333,
    milestones: list[int],
    gamma: float = 0.1,
) -> MultiStepLR:
    """
    multistep LR with warmup - used in finetunning
    """
    linear_sch = LinearLR(
        optimizer, start_factor=start_factor, total_iters=num_warmup_steps
    )
    multi_step_lr_sch = MultiStepLR(
        optimizer=optimizer,
        milestones=[m - num_warmup_steps for m in milestones],
        gamma=gamma,
    )
    return SequentialLR(
        optimizer,
        schedulers=[linear_sch, multi_step_lr_sch],
        milestones=[num_warmup_steps],
    )
