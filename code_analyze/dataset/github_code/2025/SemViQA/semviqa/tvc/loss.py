import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Focal Loss to replace CrossEntropyLoss.
        Args:
            gamma (float): Adjusts the focus on hard examples (default = 2.0).
            alpha (float): Weight for the positive class to balance (default = 0.25).
            reduction (str): 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - Output of the model.
            targets: (batch_size) - Ground truth labels.

        Returns:
            loss: Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss