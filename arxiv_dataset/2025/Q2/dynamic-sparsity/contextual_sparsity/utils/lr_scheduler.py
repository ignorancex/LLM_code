# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, SequentialLR


class LinearWarmup(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        total_iterations: int,
        min_factor: float = 1e-3,
        max_factor: float = 1.0,
        warmup_percentage: float = 0.1,
    ):
        """
        Sequential learning rate scheduler with an initial warm-up and a cool-down phases.
        """
        warmup_iterations = int(total_iterations * warmup_percentage)
        decay_iterations = total_iterations - warmup_iterations
        super(LinearWarmup, self).__init__(
            optimizer=optimizer,
            schedulers=[
                LinearLR(
                    optimizer=optimizer,
                    start_factor=min_factor,
                    end_factor=max_factor,
                    total_iters=warmup_iterations,
                ),
                LinearLR(
                    optimizer=optimizer,
                    start_factor=max_factor,
                    end_factor=min_factor,
                    total_iters=decay_iterations,
                ),
            ],
            milestones=[warmup_iterations],
        )
