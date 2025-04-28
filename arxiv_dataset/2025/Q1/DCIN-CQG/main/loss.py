import torch.nn as nn
from monai.losses import DiceLoss

class SetCriterion(nn.Module):
    # from https://github.com/Kaiseem/SLAug/blob/main/losses/__init__.py
    def __init__(self):
        super().__init__()
        self.ce_loss=nn.CrossEntropyLoss()
        self.dice_loss=DiceLoss(to_onehot_y=True,softmax=True,squared_pred=True,smooth_nr=0.0,smooth_dr=1e-6)
        self.weight_dict={'ce_loss':1, 'dice_loss':1}

    def get_loss(self, pred, gt):
        if len(gt.size())==4 and gt.size(1)==1:
            gt=gt[:,0]

        if type(pred) is not list:
            _ce=self.ce_loss(pred,gt)
            _dc=self.dice_loss(pred,gt.unsqueeze(1))
            return {'ce_loss': _ce,'dice_loss':_dc}
        else:
            ce=0
            dc=0
            for p in pred:
                ce+=self.ce_loss(p,gt)
                dc+=self.dice_loss(p,gt.unsqueeze(1))
            return {'ce_loss': ce, 'dice_loss':dc}

    def forward(self, pred, gt):
        loss_dict = self.get_loss(pred, gt)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())
        return losses

class CQGLoss(nn.Module):
    # ours
    def __init__(self, weight_geo=1.0, weight_aug=1.0, weight_diff=1.0):
        super().__init__()
        self.criterion = SetCriterion()
        self.weight_geo = weight_geo
        self.weight_aug = weight_aug
        self.weight_diff = weight_diff

    def forward(self, geo_outputs, aug_outputs, masks):
        geo_losses = self.criterion.get_loss(geo_outputs, masks)
        aug_losses = self.criterion.get_loss(aug_outputs, masks)

        geo_loss = (
            geo_losses["ce_loss"] * self.criterion.weight_dict["ce_loss"] +
            geo_losses["dice_loss"] * self.criterion.weight_dict["dice_loss"]
        )

        aug_loss = (
            aug_losses["ce_loss"] * self.criterion.weight_dict["ce_loss"] +
            aug_losses["dice_loss"] * self.criterion.weight_dict["dice_loss"]
        )

        diff_loss = nn.functional.mse_loss(geo_outputs, aug_outputs)

        total_loss = (
            self.weight_geo * geo_loss +
            self.weight_aug * aug_loss +
            self.weight_diff * diff_loss
        )

        return total_loss, geo_loss, aug_loss, diff_loss