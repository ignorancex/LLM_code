import torch
import torch.nn as nn
import pytorch_lightning as pl

# Import classification metrics with the new "task" API
from torchmetrics.functional.classification import (
    precision,
    recall,
    f1_score,
    auroc,
    average_precision,
)
# Retrieval metric (unchanged)
from torchmetrics.functional import retrieval_average_precision
from torch.optim.lr_scheduler import OneCycleLR

import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Custom modules - adjust paths as needed
from q2l_labeller.data.cutmix import CutMixCriterion,cutmix
from q2l_labeller.loss_modules.simple_asymmetric_loss import AsymmetricLoss
from q2l_labeller.loss_modules.TwoWayMultiLabelLoss import TwoWayLoss
from q2l_labeller.models.query2label import Query2Label

###############################################################################
# Utility function for custom mAP@k (ranking-based) computation
###############################################################################
def calculate_map_at_k(y_true, y_pred, k=20):
    """Calculate mAP@k for multi-label classification."""
    ap_scores = []
    for true_labels, pred_scores in zip(y_true, y_pred):
        # Sort by descending score and take top-k
        top_k_indices = pred_scores.argsort()[-k:][::-1]
        num_relevant = 0
        ap_sum = 0.0

        for i, idx in enumerate(top_k_indices):
            if true_labels[idx] == 1:
                num_relevant += 1
                ap_sum += num_relevant / (i + 1)

        if num_relevant > 0:
            ap_scores.append(ap_sum / min(num_relevant, k))
        else:
            ap_scores.append(0.0)

    return np.mean(ap_scores)

###############################################################################
# Main Lightning Module
###############################################################################
class Query2LabelTrainModule(pl.LightningModule):
    def __init__(
        self,
        data,
        backbone_desc,
        conv_out_dim,
        hidden_dim,
        num_encoders,
        num_decoders,
        num_heads,
        batch_size,
        image_dim,
        learning_rate,
        momentum,
        weight_decay,
        n_classes,
        thresh=0.5,
        use_cutmix=False,
        use_pos_encoding=False,
        use_seathru=False,
        loss="BCE",
    ):
        super().__init__()

        # Save init parameters
        self.save_hyperparameters(ignore=["model", "data"])

        # Dataloaders / data module
        self.data = data

        # Model
        self.model = Query2Label(
            model=backbone_desc,
            conv_out=conv_out_dim,
            num_classes=n_classes,
            hidden_dim=hidden_dim,
            nheads=num_heads,
            encoder_layers=num_encoders,
            decoder_layers=num_decoders,
            use_pos_encoding=use_pos_encoding,
        )

        # Loss
        if loss == "BCE":
            self.base_criterion = nn.BCEWithLogitsLoss()
        elif loss == "ASL":
            self.base_criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0)
        elif loss == "mll":
            self.base_criterion = TwoWayLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss}")

        # Optionally wrap with CutMix
        self.use_cutmix = use_cutmix
        if use_cutmix:
            self.criterion = CutMixCriterion(self.base_criterion)
        else:
            self.criterion = self.base_criterion

        # Threshold for metrics
        self.thresh = thresh

        # Number of classes
        self.n_classes = n_classes

    def forward(self, x, return_features=False):
        return self.model(x, return_features=return_features)

    ###########################################################################
    # 🚀 **Updated `evaluate()` Method with Only `ovr_` Metrics**
    ###########################################################################
    def evaluate(self, batch, stage: str = None):
        x, y = batch
        y_hat = self(x)  # raw logits

        # Compute loss
        loss = self.base_criterion(y_hat, y.float())

        # Retrieval-based AP (ranking scenario)
        rmap = retrieval_average_precision(y_hat, y.long())

        # Compute overall (micro) classification metrics
        ovr_prec = precision(y_hat, y.long(), task="multilabel", num_labels=self.n_classes, average="micro", threshold=self.thresh)
        ovr_recall = recall(y_hat, y.long(), task="multilabel", num_labels=self.n_classes, average="micro", threshold=self.thresh)
        ovr_f1 = f1_score(y_hat, y.long(), task="multilabel", num_labels=self.n_classes, average="micro", threshold=self.thresh)

        # Compute AUROC & mAP
        roc_auc = auroc(y_hat, y.long(), task="multilabel", num_labels=self.n_classes, average="micro")
        mAP = average_precision(y_hat, y.long(), task="multilabel", num_labels=self.n_classes, average="micro")

        # Custom mAP@20
        y_hat_np = y_hat.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        map_20 = calculate_map_at_k(y_np, y_hat_np, k=20)

        # Log Metrics
        if stage is not None:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_rmap", rmap, prog_bar=True)
            self.log(f"{stage}_ovr_prec", ovr_prec, prog_bar=True)
            self.log(f"{stage}_ovr_recall", ovr_recall, prog_bar=True)
            self.log(f"{stage}_ovr_f1", ovr_f1, prog_bar=True)
            self.log(f"{stage}_roc_auc", roc_auc, prog_bar=True)
            self.log(f"{stage}_mAP", mAP, prog_bar=True)
            self.log(f"{stage}_mAP@20", map_20, prog_bar=True)

    ###########################################################################
    # Training Step
    ###########################################################################
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)

    #     if self.use_cutmix:
    #         if isinstance(y, tuple) and len(y) == 3:
    #             loss = self.CutMixCriterion(y_hat, y)
    #         else:
    #             raise ValueError(f"❌ CutMix Expected (y1, y2, lam), but got {y}")
    #     else:
    #         loss = self.criterion(y_hat, y.float())

    #     self.log("train_loss", loss)
    #     return loss
    def training_step(self, batch, batch_idx):
        x, y = batch

        # Apply CutMix augmentation conditionally
        if self.use_cutmix:
            # Apply CutMix augmentation (returns data and targets tuple)
            x_mixed, targets = cutmix((x, y), alpha=1.0)  # Adjust alpha as needed
            y_hat = self(x)

            loss = self.criterion(y_hat, targets)  # targets is (y1, y2, lam)

        else:
            # Standard (no CutMix)
            y_hat = self(x)
            loss = self.base_criterion(y_hat, y.float())

        self.log("train_loss", loss)
        return loss



    ###########################################################################
    # Validation Step
    ###########################################################################
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    ###########################################################################
    # Test Step
    ###########################################################################
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    ###########################################################################
    # Optimizer and Learning Rate Scheduler
    ###########################################################################
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        try:
            steps_per_epoch = int(len(self.data.train_dataloader()))  # Ensure it's an integer
        except TypeError:
            raise ValueError("❌ Error: Could not determine steps_per_epoch, check train_dataloader().")

        lr_scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                anneal_strategy="cos",
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
