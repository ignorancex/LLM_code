import torch
from torch import nn


class Loss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UnsupervisedLoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((1 - target) * output)


class SupervisedLoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # y_positive = -torch.log1p(-torch.exp(-output))
        y_positive = -torch.log(1 - torch.exp(-output) + 1e-6)
        y_unlabeled = output
        return torch.mean((1 - target) * y_unlabeled + target * y_positive)


class PUBaseLoss(Loss):
    def __init__(self, beta: float, use_abs: bool = True):
        super().__init__()
        self.beta = beta
        self.use_abs = use_abs

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        positive = target
        unlabeled = 1 - target

        n_positive = torch.sum(positive).clamp_min(1)
        n_unlabeled = torch.sum(unlabeled).clamp_min(1)

        y_positive = self.positive_loss(output)
        y_unlabeled = self.unlabeled_loss(output)

        positive_risk = torch.sum(self.beta * positive * y_positive / n_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.beta * positive / n_positive) * y_unlabeled)

        # use abs operator following absPU
        if self.use_abs:
            return positive_risk + torch.abs(negative_risk)

        # use max operator following nnPU
        if negative_risk < 0:
            return -1 * negative_risk
        else:
            return positive_risk + negative_risk

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PULoss(PUBaseLoss):
    def __init__(self, beta: float):
        super().__init__(beta=beta)

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        # return -torch.log1p(-torch.exp(-x))
        return -torch.log(1 - torch.exp(-x) + 1e-6)

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x


def load(algorithm: str, beta: float) -> Loss:
    if algorithm == "Unsupervised":
        return UnsupervisedLoss()
    elif algorithm == "Supervised":
        return SupervisedLoss()
    else:
        return PULoss(beta=beta)
