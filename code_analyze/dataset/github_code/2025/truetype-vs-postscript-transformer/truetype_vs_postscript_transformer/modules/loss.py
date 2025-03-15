"""Provides a hybrid loss function for classification and regression tasks."""

import torch
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    """A loss function that combines classification, regression, and Chamfer losses."""

    def __init__(
        self,
        ignore_index: int = -100,
        ce_weight: float = 1.0,
        mse_weight: float = 1.0,
        chamfer_weight: float = 1.0,
        n_samples: int = 4,
    ) -> None:
        """Initialize the loss function with specified parameters."""
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.chamfer_weight = chamfer_weight

        self.register_buffer(
            "t",
            torch.linspace(0, 1, steps=n_samples).view(1, -1, 1),
        )

        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Compute the hybrid loss for classification, regression, and Chamfer tasks."""
        logits, outputs = predictions
        labels, regression_targets = targets

        logits = logits.flatten(end_dim=-2)
        labels = labels.flatten()
        outputs = outputs.flatten(end_dim=-2)
        regression_targets = regression_targets.flatten(end_dim=-2)

        valid_mask = labels != self.ignore_index

        ce_loss = (
            self.ce(logits, labels)
            if self.ce_weight != 0
            else torch.tensor(0.0, device=outputs.device)
        )
        mse_loss = (
            self._compute_mse(outputs, regression_targets, valid_mask)
            if self.mse_weight != 0
            else torch.tensor(0.0, device=outputs.device)
        )
        chamfer_loss = (
            self._compute_chamfer(outputs, regression_targets, valid_mask)
            if self.chamfer_weight != 0
            else torch.tensor(0.0, device=outputs.device)
        )

        return (
            self.ce_weight * ce_loss
            + self.mse_weight * mse_loss
            + self.chamfer_weight * chamfer_loss
        )

    def _compute_mse(
        self,
        outputs: Tensor,
        regression_targets: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Compute masked mean squared error loss."""
        outputs_valid = outputs[valid_mask]
        regression_targets_valid = regression_targets[valid_mask]
        return self.mse(outputs_valid, regression_targets_valid)

    def _compute_chamfer(
        self,
        outputs: Tensor,
        regression_targets: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Compute Chamfer loss with sampled points along curves for a batch."""
        valid_outputs = outputs[valid_mask]
        valid_targets = regression_targets[valid_mask]

        sampled_outputs = self._batch_sample_bezier_points(valid_outputs)
        sampled_targets = self._batch_sample_bezier_points(valid_targets)

        distances = torch.cdist(sampled_outputs, sampled_targets, p=2)
        forward_loss = distances.min(dim=1).values.mean()
        backward_loss = distances.min(dim=0).values.mean()

        return forward_loss + backward_loss

    def _batch_sample_bezier_points(
        self,
        coords: Tensor,
    ) -> Tensor:
        """Batch sample n points along quadratic Bézier curves."""
        p1 = coords[:, 0:2].unsqueeze(1)
        p2 = coords[:, 2:4].unsqueeze(1)
        p3 = coords[:, 4:6].unsqueeze(1)

        p0 = torch.roll(p3, shifts=1, dims=0)

        sampled_points = (
            (1 - self.t) ** 3 * p0
            + 3 * (1 - self.t) ** 2 * self.t * p1
            + 3 * (1 - self.t) * self.t**2 * p2
            + self.t**3 * p3
        )

        return sampled_points.view(-1, 2)


class KLDivergenceLoss(nn.Module):
    """KL Divergence Loss for VAE."""

    def __init__(self) -> None:
        """Initialize the KL Divergence loss function."""
        super().__init__()

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute the KL divergence loss for VAE."""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss.mean()
