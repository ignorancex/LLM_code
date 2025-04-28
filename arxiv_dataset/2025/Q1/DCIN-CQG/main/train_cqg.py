import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchmetrics

from loss import SetCriterion, CQGLoss

# Lightning model wrapper
class CQGSegmentationModel(pl.LightningModule):
    def __init__(self, backbone="efficientnet-b2", lr=0.001, weight_geo=1.0, weight_aug=1.0, weight_diff=1.0):
        super(CQGSegmentationModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=3,
            classes=5,
            activation=None
        )

        self.cqg_loss = CQGLoss(weight_geo, weight_aug, weight_diff)
        self.criterion = SetCriterion()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # prepare inputs
        geo_images, aug_images, masks = batch
        geo_images = geo_images.to(dtype=torch.float32)
        aug_images = aug_images.to(dtype=torch.float32)
        masks = masks.to(dtype=torch.long)

        # forward pass
        geo_outputs = self(geo_images)
        aug_outputs = self(aug_images)

        # compute losses
        total_loss, geo_loss, aug_loss, diff_loss = self.cqg_loss(geo_outputs, aug_outputs, masks)

        # metrics
        geo_preds = torch.argmax(geo_outputs, dim=1)
        aug_preds = torch.argmax(aug_outputs, dim=1)
        dice_geo = torchmetrics.functional.dice(geo_preds, masks, num_classes=5, average='macro')
        dice_aug = torchmetrics.functional.dice(aug_preds, masks, num_classes=5, average='macro')

        # logging
        self.log("train_geo_loss", geo_loss, on_epoch=True, prog_bar=True)
        self.log("train_aug_loss", aug_loss, on_epoch=True, prog_bar=True)
        self.log("train_diff_loss", diff_loss, on_epoch=True, prog_bar=True)
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("train_dice_geo", dice_geo, on_epoch=True, prog_bar=True)
        self.log("train_dice_aug", dice_aug, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # prepare inputs
        images, masks = batch
        images = images.to(dtype=torch.float32)
        masks = masks.to(dtype=torch.long)

        # forward pass
        outputs = self(images)

        # loss
        losses = self.criterion.get_loss(outputs, masks)
        total_loss = (
            losses["ce_loss"] * self.criterion.weight_dict["ce_loss"] +
            losses["dice_loss"] * self.criterion.weight_dict["dice_loss"]
        )

        # metrics
        preds = torch.argmax(outputs, dim=1)
        dice_score = torchmetrics.functional.dice(preds, masks, num_classes=5, average='macro')

        # logging
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice_score, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
