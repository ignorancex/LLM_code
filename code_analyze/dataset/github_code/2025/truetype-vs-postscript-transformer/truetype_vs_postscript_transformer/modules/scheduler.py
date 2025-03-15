"""Learning rate schedulers for training the model."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupDecayLR(LambdaLR):
    """Learning rate scheduler with warm-up phase followed by decay phase.

    Args:
        optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
        warmup_steps (int): Number of warm-up steps.
        total_steps (int): Total number of training steps.
        last_epoch (int): The index of the last epoch. Default: -1.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        """Initialize a new instance of WarmupDecayLR."""

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            return (warmup_steps / current_step) ** 0.5

        super().__init__(optimizer, lr_lambda, last_epoch)
