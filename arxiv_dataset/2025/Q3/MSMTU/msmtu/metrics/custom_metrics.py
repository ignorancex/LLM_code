import torch
from torchmetrics import Metric


def _check_same_shape(preds: torch.Tensor, target: torch.Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )


class MSE(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
            self,
            squared: bool = True,
            averaged: bool = True,
            num_classes: int = 1,
    ):
        super().__init__()
        # "sum_squared_error" state will be used to accumulate the sum of squared errors, and it is initialized to 0 of size num classes
        self.add_state("sum_squared_error", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")
        self.squared = squared
        self.averaged = averaged

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        _check_same_shape(preds, target)
        self.sum_squared_error += torch.sum((preds - target) ** 2, dim=0)
        self.n_obs += target.shape[0]

    def compute(self):
        """ Computes the mean squared error over state.
        - If self.squared is False, then the square root is taken.
        - If self.average is "macro", then the mean is taken over the number of classes.
        """
        mse = self.sum_squared_error / self.n_obs
        if not self.squared:
            mse = mse.sqrt()
        return mse.mean() if self.averaged else mse


class MAE(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
            self,
            averaged: bool = True,
            absolute: bool = True,
            num_classes: int = 1,
    ):
        super().__init__()
        # "sum_squared_error" state will be used to accumulate the sum of squared errors, and it is initialized to 0 of size num classes
        self.add_state("sum_error", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")
        self.averaged = averaged
        self.absolute = absolute

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        _check_same_shape(preds, target)
        # If self.absolute is True, then the absolute value is taken in self.sum_absolute_error state
        self.sum_error += torch.sum(torch.abs(preds - target) if self.absolute else (preds - target), dim=0)
        self.n_obs += target.shape[0]

    def compute(self):
        """ Computes the mean squared error over state.
        - If self.squared is False, then the square root is taken.
        - If self.average is "macro", then the mean is taken over the number of classes.
        """
        me = self.sum_error / self.n_obs
        return me.mean() if self.averaged else me

