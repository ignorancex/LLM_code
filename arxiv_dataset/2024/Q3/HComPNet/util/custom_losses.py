import torch 
import torch.nn.functional as F
from kornia.losses import FocalLoss

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, weights):
        batch_loss = F.cross_entropy(inputs, targets, reduction='none')
        class_mask = F.one_hot(targets, num_classes=inputs.size(1)).type(inputs.dtype)
        weighted_losses = (batch_loss.unsqueeze(1) * class_mask * weights).sum(dim=1)
        loss = weighted_losses.mean()
        return loss


class WeightedNLLLoss(torch.nn.Module):
    def __init__(self, device):
        super(WeightedNLLLoss, self).__init__()
        self.device = device

    def forward(self, logits, targets, weights=None):
        log_probs = F.log_softmax((logits),dim=1)
        batch_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        if weights is not None:
            weights = weights.to(self.device)
            class_mask = F.one_hot(targets, num_classes=log_probs.size(1)).type(log_probs.dtype).to(self.device)
            weighted_losses = (batch_loss.unsqueeze(1) * class_mask * weights).sum(dim=1)
            loss = weighted_losses.mean()
        else:
            loss = batch_loss.mean()

        return loss


class FocalLossWrapper(torch.nn.Module):
    def __init__(self, device, alpha, gamma=2.0, reduction='mean', weight=None):
        super(FocalLossWrapper, self).__init__()
        self.loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction).to(device)

    # weights is here because it is used by other custom losses it will not be used
    def forward(self, pred, target, weights=None):
        return self.loss.forward(pred, target)
