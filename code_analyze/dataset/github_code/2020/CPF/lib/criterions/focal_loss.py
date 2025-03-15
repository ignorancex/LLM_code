# this file come from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
import torch
from torch.nn import functional as F
import numpy as np


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor = None,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if masks is None:
        masks = torch.ones_like(inputs)
    masks.requires_grad_ = False

    if masks.sum().detach().cpu().item() != 0:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # TENSOR (B, N)

        # NOTE: apply masks, which is the combinations of
        # recov_contact_in_image_mask  &  the CollateQueries.PADDING_MASK
        ce_loss = masks * ce_loss

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t)**gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            # the loss influenced (set zero) by the masks.
            # It is not reasonable to directly apply loss.mean()
            loss = loss.sum() / masks.sum()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
    else:
        return torch.Tensor([0.0]).float().to(inputs.device)


def multiclass_focal_loss(
    inputs: torch.Tensor,  # TENSOR (B x N, C)
    targets: torch.Tensor,  # TENSOR (B x N, C)
    masks: torch.Tensor,  # TENSOR (B x N, 1)
    alpha=None,  # TENSOR (C, 1)
    gamma: float = 2,
    reduction: str = "none",
):
    # first check whether mask out every element
    if masks.sum().detach().cpu().item() != 0:
        n_classes = inputs.shape[1]
        logit = F.softmax(inputs, dim=1)  # TENSOR (B x N, C)
        if alpha is None:
            alpha = torch.ones((n_classes), requires_grad=False)

        if alpha.device != inputs.device:
            alpha = alpha.to(inputs.device)

        epsilon = 1e-10
        pt = torch.sum((targets * logit), dim=1, keepdim=True) + epsilon  # TENSOR (B x N, 1)
        log_pt = pt.log()  # TENSOR (B x N, 1)

        # NOTE: here add alpha
        # TENSOR (B x N, C) -> TENSOR (B x N, 1), retrive back the idx value from one_hot targets
        targets_idx = torch.argmax(targets, dim=1, keepdim=True).long()  # TENSOR (B x N, 1)
        alpha = alpha[targets_idx]  # TENSOR ( B x N, 1)

        focal_loss = -1 * alpha * (torch.pow((1 - pt), gamma) * log_pt)  # TENSOR (B x N, 1)
        masked_focal_loss = focal_loss * masks  # TENSOR (B x N, 1)

        if reduction == "mean":
            loss = masked_focal_loss.sum() / masks.sum()
        elif reduction == "sum":
            loss = masked_focal_loss.sum()
        else:
            loss = masked_focal_loss

        return loss
    else:
        return torch.Tensor([0.0]).float().to(inputs.device)


class MultiFocalLoss(torch.nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError("Not support alpha type")

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
