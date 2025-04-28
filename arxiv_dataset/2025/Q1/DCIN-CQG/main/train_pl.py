import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchmetrics

from loss import SetCriterion


# Lightning model wrapper
class SegmentationModel(pl.LightningModule):
    def __init__(self, backbone="efficientnet-b2", lr=0.001):
        super(SegmentationModel, self).__init__()
        self.model = smp.Unet(
                        encoder_name=backbone,
                        encoder_weights='imagenet',
                        in_channels=3,
                        classes=5,
                        activation=None
                    )
        self.criterion = SetCriterion()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # prepare inputs
        images, masks = batch
        images = images.to(dtype=torch.float32)
        masks = masks.to(dtype=torch.long)

        # forward pass
        outputs = self(images)

        # loss
        losses = self.criterion.get_loss(outputs, masks)
        total_loss = (
            losses["ce_loss"] * self.criterion.weight_dict["ce_loss"]
            + losses["dice_loss"] * self.criterion.weight_dict["dice_loss"]
        )

        # metrics
        preds = torch.argmax(outputs, dim=1)
        dice_score = torchmetrics.functional.dice(preds, masks, num_classes=5, average='macro')

        # logging
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("train_dice", dice_score, on_epoch=True, prog_bar=True)
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
            losses["ce_loss"] * self.criterion.weight_dict["ce_loss"]
            + losses["dice_loss"] * self.criterion.weight_dict["dice_loss"]
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
