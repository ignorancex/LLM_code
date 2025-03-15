"""PyTorch Lightning Module for training FontClassifier models."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import Tensor

from truetype_vs_postscript_transformer.models import (
    T3,
    ClsTokenFontClassifier,
)
from truetype_vs_postscript_transformer.modules.mask import (
    classification_mask,
    create_mask,
)
from truetype_vs_postscript_transformer.modules.scheduler import WarmupDecayLR


class StyleClassifierLM(pl.LightningModule):
    """A PyTorch Lightning Module for training FontClassifier models."""

    def __init__(
        self,
        model: Literal["t3", "cls"],
        num_layers: int,
        emb_size: int,
        nhead: int,
        class_labels: list[str],
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        lr: float = 0.001,  # noqa: ARG002
        warmup_steps: int = 250,  # noqa: ARG002
        outline_format: Literal["truetype", "postscript"] = "postscript",
    ) -> None:
        """Initialize the module with the given hyperparameters.

        Args:
            model: Type of model to use for classification.
            outline_format: Format of the font outlines.
            num_layers: Number of encoder layers in the model.
            emb_size: Embedding size for the model.
            nhead: Number of attention heads.
            class_labels: List of class labels for the classifier.
            dim_feedforward: Dimension of the feedforward layer.
            dropout: Dropout rate.
            lr: Learning rate for the optimizer.
            warmup_steps: Number of warm-up steps for the scheduler.

        """
        super().__init__()
        self.save_hyperparameters()

        if model == "t3":
            self.model = T3(
                num_layers=num_layers,
                emb_size=emb_size,
                nhead=nhead,
                num_classes=len(class_labels),
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

        elif model == "cls":
            self.model = ClsTokenFontClassifier(
                num_layers=num_layers,
                emb_size=emb_size,
                nhead=nhead,
                num_classes=len(class_labels),
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                outline_format=outline_format,
            )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.test_predictions = []
        self.test_targets = []

    def forward(
        self,
        src: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Perform a forward pass through the FontClassifier."""
        if self.model == "t3":
            src_mask, src_padding_mask = create_mask(
                glyph_tensor=src,
                causal=False,
                padding_token=-1,
            )
        else:
            src_mask, src_padding_mask = classification_mask(src)

        return self.model(
            src=src,
            src_mask=src_mask,
            src_padding_mask=src_padding_mask,
        )

    def training_step(
        self,
        batch: tuple[tuple[Tensor, Tensor], Tensor, Tensor],
        _batch_idx: int,
    ) -> Tensor:
        """Execute a training step."""
        glyph, codepoint, font_index = batch
        predictions = self(glyph)

        loss = self.loss_fn(predictions, font_index)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )
        return loss

    def validation_step(
        self,
        batch: tuple[tuple[Tensor, Tensor], Tensor, Tensor],
        _batch_idx: int,
    ) -> None:
        """Execute a validation step."""
        glyph, codepoint, font_index = batch
        predictions = self(glyph)
        loss = self.loss_fn(predictions, font_index)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )

    def test_step(
        self,
        batch: tuple[tuple[Tensor, Tensor], Tensor, Tensor],
        _batch_idx: int,  # noqa: PT019
    ) -> None:
        """Execute a test step."""
        glyph, codepoint, font_index = batch
        predictions = self(glyph)
        loss = self.loss_fn(predictions, font_index)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )

        predicted_labels = torch.argmax(predictions, dim=1)
        self.test_predictions.extend(predicted_labels.cpu().numpy())
        self.test_targets.extend(font_index.cpu().numpy())

    def on_test_epoch_end(self) -> None:
        """Generate and save confusion matrix at the end of testing."""
        if self.trainer.global_rank == 0 and self.trainer.log_dir:
            log_dir = Path(self.trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            report: str = classification_report(
                self.test_targets,
                self.test_predictions,
                target_names=self.hparams.class_labels,  # type: ignore[attr-defined]
                digits=3,
                zero_division=0,
            )

            with Path.open(log_dir / "classification_report.txt", "w") as f:
                f.write(report)

            cm = confusion_matrix(self.test_targets, self.test_predictions)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=self.hparams.class_labels,  # type: ignore[attr-defined]
            )

            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(ax=ax, cmap="viridis", colorbar=False)

            ax.grid(visible=False)

            ax.set_title("Confusion Matrix", fontsize=14, pad=20)
            ax.set_xlabel("Predicted Labels", fontsize=12, labelpad=10)
            ax.set_ylabel("True Labels", fontsize=12, labelpad=10)

            plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=10)
            plt.setp(ax.get_yticklabels(), fontsize=10)

            cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)

            plt.tight_layout()
            plt.savefig(log_dir / "confusion_matrix.png", dpi=300)
            plt.savefig(log_dir / "confusion_matrix.pdf")
            plt.close(fig)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore[attr-defined]
        )
        scheduler: LRSchedulerConfigType = {
            "scheduler": WarmupDecayLR(
                optimizer,
                warmup_steps=self.hparams.warmup_steps,  # type: ignore[attr-defined]
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
