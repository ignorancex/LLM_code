import torch
from torch import nn, Tensor

# In-depth discussion of loss functions: https://arxiv.org/pdf/2211.02989.pdf


def LogMSE_loss(pred: Tensor, target: Tensor) -> Tensor:
    return nn.MSELoss()(torch.log(abs(pred) + 1), torch.log(abs(target) + 1))


def NMSE_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Computes the mean squared error normalized by target.
    """
    return nn.MSELoss(reduction='mean')(pred, target) / torch.mean(target**2)


def NMSE_Sqr_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Computes the mean squared error normalized by target. The square of predicted
    values is used.
    """
    return nn.MSELoss(reduction='mean')(pred**2, target**2) / torch.mean(target**4)


def NMSE_Abs_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Computes the mean squared error normalized by target. The absolute of predicted
    values is used.
    """
    return nn.MSELoss(reduction='mean')(torch.abs(pred), target) / torch.mean(target**2)


def MAPE_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean absolute error loss function, MAPE = MEAN(ABS(y-y_true)/y_true)

    :param Tensor pred: predicted value
    :param Tensor target: target value
    :return Tensor: computed error
    """
    return torch.mean(abs(pred - target) / target) * 100


def RRMSE_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((pred - target)**2) / torch.sum(pred**2))
