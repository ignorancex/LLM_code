import torch
import torch.nn as nn
import torch.nn.functional as F


class _TotalVarLoss3d(nn.Module):

    def _calculate_variation(self, x, dim):
        assert dim in (2, 3, 4)

        length = x.size(dim)
        x0 = torch.index_select(x, dim, torch.arange(length - 1))
        x1 = torch.index_select(x, dim, torch.arange(1, length))

        return x0, x1

    def forward(self, output):
        x0, x1 = self._calculate_variation(output, 2)
        y0, y1 = self._calculate_variation(output, 3)
        z0, z1 = self._calculate_variation(output, 4)
        total_var = F.l1_loss(x0, x1) + F.l1_loss(y0, y1)\
            + F.l1_loss(z0, z1)

        return total_var


class JointLoss(nn.Module):

    def __init__(self, w_ce, w_var):
        super().__init__()

        self.w_ce = w_ce
        self.w_var = w_var
        self.ce_fn = nn.CrossEntropyLoss()
        self.var_fn = _TotalVarLoss3d()

    def forward(self, output, target):
        ce_loss = self.ce_fn(output, target)
        var_loss = self.var_fn(output)
        joint_loss = self.w_ce * ce_loss + self.w_var * var_loss

        return joint_loss


class DiceLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, output, target):
        eps = 1e-8
        fg_output = torch.softmax(output, dim=1)[:, 1:, ...]
        fg_target = F.one_hot(target, self.num_classes)[..., 1:]
        fg_target = fg_target.permute(0, 4, 1, 2, 3)
        dice_loss = 1 - 2 * (fg_output * fg_target).sum()\
            / (fg_output.sum() + fg_target.sum() + eps)

        return dice_loss


class SegLoss(nn.Module):

    def __init__(self, w_ce, w_dice, num_classes):
        super().__init__()

        self.w_ce = w_ce
        self.w_dice = w_dice
        self.ce_fn = nn.CrossEntropyLoss()
        self.dice_fn = DiceLoss(num_classes)

    def forward(self, output, target):
        ce_loss = self.ce_fn(output, target)
        dice_loss = self.dice_fn(output, target)
        total_loss = self.w_ce * ce_loss + self.w_dice * dice_loss

        return total_loss
