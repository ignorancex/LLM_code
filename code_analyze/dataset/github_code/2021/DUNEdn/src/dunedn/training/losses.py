"""
    This module implements several losses.
    
    Main option is reduction, which could be either `mean` (default) or `None`.
"""
import torch
import numpy as np
from torch import nn
from abc import ABC, abstractmethod
from dunedn.training.ssim import stat_ssim
from dunedn.utils.utils import confusion_matrix


EPS = torch.Tensor([torch.finfo(torch.float64).eps])


class Loss(ABC):
    """Abstract loss function class."""

    def __init__(self, a=0.5, data_range=1.0, reduction="mean"):
        """
        Parameters
        ----------
        a: float
            Relative weight of the loss constributions.
        data_range: float
            Data interval.
        reduction: str
            Available options mean | none.
        """
        self.a = a
        self.data_range = data_range
        self.reduction = reduction

    @abstractmethod
    def __call__(self, y_pred, y_true):
        """Compute the loss function"""
        pass


class LossMse(Loss):
    """Mean squared error loss function.

    Computes ``L2(y_true, y_pred) = mean((y_true - y_pred)**2)``.
    """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super(LossMse, self).__init__(reduction=reduction)
        self.loss = nn.MSELoss(reduction="none")
        self.name = "MSE"

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        loss = self.loss(y_pred, y_true)
        if self.reduction == "mean":
            return loss.mean()
        return loss.mean([1, 2, 3])


class LossImae(Loss):
    """Mean absolute error on integrated charge loss function.

    Computes ``IMAE(y_true, y_pred) = mean(|y_true.sum(-1) - y_pred.sum(-1)|)``.
    """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super(LossImae, self).__init__(reduction=reduction)
        self.loss = nn.L1Loss(reduction="none")
        self.name = "IMAE"

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        loss = self.loss(y_pred.sum(-1), y_true.sum(-1))
        if self.reduction == "mean":
            return loss.mean()
        return loss.reshape([loss.shape[0], -1]).mean(-1)


class LossSsim(Loss):
    """Statistical structural similarity loss function.

    Computes ``Lssim(y_true, y_pred) = 1 - stat-ssim(y_true, y_pred)``.
    """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super(LossSsim, self).__init__(a, data_range, reduction)
        self.name = "Lssim"

    def __call__(self, x_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        return 1 - stat_ssim(
            x_pred, y_true, data_range=self.data_range, reduction=self.reduction
        )


class Ssim(Loss):
    """Statistical structural similarity function.

    Computes ``ssim(y_true, y_pred) = stat-ssim(y_true, y_pred)``.
    """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super(Ssim, self).__init__(a, data_range, reduction)
        self.name = "ssim"

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        return stat_ssim(
            y_pred, y_true, data_range=self.data_range, reduction=self.reduction
        )


class LossSsimL2(Loss):
    """Stat ssim + MSE loss function.

    Computes ``(a * L2 + (1 - a) * 1e-3 * Lssim)(y_true, y_pred)``.
    """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super(LossSsimL2, self).__init__(a, data_range, reduction)
        self.name = "ssim_l2"

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        loss1 = nn.MSELoss(reduction=self.reduction)(y_pred, y_true)
        if self.reduction == "none":
            loss1 = loss1.mean([1, 2, 3])
        loss2 = 1 - stat_ssim(
            y_pred, y_true, data_range=self.data_range, reduction=self.reduction
        )
        return self.a * loss1 + (1 - self.a) * 1e-3 * loss2


class LossSsimL1(Loss):
    """Stat ssim + mean absolute error loss function.

    Computes ``(a * L1 + (1 - a) * 1e-3 * Lssim)(y_true, y_pred)``.
    """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super(LossSsimL1, self).__init__(a, data_range, reduction)
        self.name = "ssim_l1"

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        loss1 = (y_pred - y_true).abs()
        if self.reduction == "mean":
            loss1 = loss1.mean()
        elif self.reduction == "none":
            loss1 = loss1.mean([1, 2, 3])
        loss2 = 1 - stat_ssim(
            y_pred, y_true, data_range=self.data_range, reduction=self.reduction
        )
        return self.a * loss1 + (1 - self.a) * 1e-3 * loss2


class LossBce(Loss):
    """Binary cross entropy loss function.

    Computes ``Xent(y_true, y_pred)``.
    """

    def __init__(self, ratio=0.5, reduction="mean"):
        """
        Ratio is the number of positive against negative example in training
        set. It's used for reweighting the cross entropy

        Parameters
        ----------
        reduction: str
            Available options mean | sum | none.
        """
        super(LossBce, self).__init__(0, 0, reduction)
        self.ratio = ratio
        self.name = "xent"

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        log = lambda x: torch.log(x + EPS.to(x.device))
        loss = -y_true * log(y_pred) / self.ratio - (1 - y_true) * log(1 - y_pred) / (
            1 - self.ratio
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LossSoftDice(Loss):
    """Soft dice loss function."""

    def __init__(self, reduction="mean"):
        """
        Reduction: str
            'mean' | 'none'
        """
        super(LossSoftDice, self).__init__(0, 0, reduction)
        self.name = "softdice"

    def dice(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        eps = EPS.to(x.device)
        ix = 1 - x
        iy = 1 - y
        num1 = (x * y).sum((-1, -2)) + eps
        den1 = (x * x + y * y).sum((-1, -2)) + eps
        num2 = (ix * iy).sum((-1, -2)) + eps
        den2 = (ix * ix + iy * iy).sum((-1, -2)) + eps
        return num1 / den1 + num2 / den2

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        ratio = self.dice(y_pred, y_true)
        loss = 1 - ratio
        if self.reduction == "mean":
            return loss.mean()
        return loss


class LossBceDice(Loss):
    """Binary xent + soft dice loss function."""

    def __init__(self, ratio=0.5, reduction="mean"):
        """
        Reduction: str
            'mean' | 'none'
        """
        super(LossBceDice, self).__init__(0, 0, reduction)
        self.bce = LossBce(ratio, reduction="none")
        self.dice = LossSoftDice(reduction="none")
        self.name = "xent_softdice"

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        shape = [y_pred.shape[0], -1]
        bce = self.bce(y_pred, y_true).reshape(shape).mean(-1)
        dice = -torch.log(self.dice.dice(y_pred, y_true))
        loss = bce + dice
        if self.reduction == "mean":
            return loss.mean()
        return loss


class LossPsnr(Loss):
    """Peak signal to noise ration function."""

    def __init__(self, reduction="mean"):
        super(LossPsnr, self).__init__(reduction=reduction)
        self.mse = nn.MSELoss(reduction="none")
        self.name = "psnr"

    def __call__(self, y_noisy, y_clear):
        """
        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted tensor, of shape=(N,C,W,H).
        y_true: torch.Tensor
            Target tensor, of shape=(N,C,W,H).
        """
        nimages = y_clear.shape[0]
        x1 = y_clear.reshape(nimages, -1)
        x2 = y_noisy.reshape(nimages, -1)
        mse = self.mse(x1, x2).mean(-1)
        m2 = x1.max(-1).values ** 2
        zero = torch.Tensor([0.0]).to(x1.device)
        eps = EPS.to(x1.device)
        psnr = torch.where(m2 == 0, zero, 10 * torch.log10(m2 / (mse + eps)))
        if self.reduction == "none":
            return psnr
        return psnr.mean()


class LossCfnm(Loss):
    """Confusion matrix function."""

    def __init__(self, reduction="mean"):
        self.name = "cfnm"

    def __call__(self, y_pred, y_true):
        # compute the confusion matrix from cuda tensors
        n = len(y_pred)
        os = y_pred.cpu().numpy().reshape([n, -1])
        ts = y_true.cpu().numpy().reshape([n, -1])
        cfnm = []
        for o, t in zip(os, ts):
            hit = o[t.astype(bool)]
            no_hit = o[~t.astype(bool)]
            cfnm.append(confusion_matrix(hit, no_hit, 0.5))
        cfnm = np.stack(cfnm)

        cfnm = cfnm / cfnm[0, :].sum()
        tp = [cfnm[:, 0].mean(), cfnm[:, 0].std() / np.sqrt(n)]
        fp = [cfnm[:, 1].mean(), cfnm[:, 1].std() / np.sqrt(n)]
        fn = [cfnm[:, 2].mean(), cfnm[:, 2].std() / np.sqrt(n)]
        tn = [cfnm[:, 3].mean(), cfnm[:, 3].std() / np.sqrt(n)]

        return tp, fp, fn, tn


def get_loss(loss):
    """Utility function to retrieve loss from loss name.

    Parameters
    ----------
    loss: str
        Available options:

        - mse
        - imae
        - ssim
        - ssim_l2
        - ssim_l1
        - bce
        - softdice
        - cfnm

    Returns
    -------
    Loss
        The query loss class.

    Raises
    ------
    NotImplementedError
        If `loss` is not in:

        - mse
        - imae
        - ssim
        - ssim_l2
        - ssim_l1
        - bce
        - softdice
        - cfnm
    """
    if loss == "mse":
        return LossMse
    elif loss == "imae":
        return LossImae
    elif loss == "ssim":
        return LossSsim
    elif loss == "ssim_l2":
        return LossSsimL2
    elif loss == "ssim_l1":
        return LossSsimL1
    elif loss == "bce":
        return LossBce
    elif loss == "softdice":
        return LossSoftDice
    elif loss == "bce_dice":
        return LossBceDice
    elif loss == "psnr":
        return LossPsnr
    elif loss == "cfnm":
        return LossCfnm
    else:
        raise NotImplementedError("Loss function not implemented")


def get_metric(metric):
    """Utility function to retrieve metrics from loss name.

    Parameters
    ----------
    metric: str
        Available options:

        - mse
        - imae
        - ssim
        - psnr
        - bce
        - softdice
        - cfnm

    Returns
    -------
    Loss
        The query class.

    Raises
    ------
    NotImplementedError
        If `loss` is not in:

        - mse
        - imae
        - ssim
        - bce
        - softdice
        - cfnm
    """
    if metric == "mse":
        return LossMse
    elif metric == "imae":
        return LossImae
    elif metric == "ssim":
        return Ssim
    elif metric == "psnr":
        return LossPsnr
    elif metric == "xent":
        return LossBce
    elif metric == "softdice":
        return LossSoftDice
    elif metric == "cfnm":
        return LossCfnm
    else:
        raise NotImplementedError("Loss function not implemented")
