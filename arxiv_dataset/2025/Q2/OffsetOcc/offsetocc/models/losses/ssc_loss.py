"""Semantic Scene Completion loss implementation.

Adapted from
    https://github.com/astra-vision/MonoScene/blob/master/monoscene/loss/ssc_loss.py
"""

# Python Libraries
import torch
from torch import nn
import torch.nn.functional as F

# Local modules
from offsetocc.registry import MODELS


@MODELS.register_module()
class SSCLoss(nn.Module):
    """Semantic Scene Completion loss.

    A loss combining the geometrical voxel occupancy and the semantic class
    occupancy.

    Attributes
    __________
    alpha: semantic completion loss balancer
    beta: geometric completion loss balancer
    """

    def __init__(
            self,
            free_class_index: int = 17,
            alpha: float = 1.0,
            beta: float = 1.0,
            loss_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.free_class_index = free_class_index
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

    def _geo_scal_loss(self, pred, ssc_target) -> torch.Tensor:
        """Compute the geometrical occupancy loss.

        Computes the loss on the class agnostic voxel occupancy.
        """
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, self.free_class_index, :, :, :]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != self.free_class_index

        batch_precision = []
        batch_recall = []
        batch_spec = []

        for pred_i, target_i, mask_i, nonempty_target_i, nonempty_probs_i, empty_probs_i in zip(
                pred, ssc_target, mask, nonempty_target, nonempty_probs, empty_probs
        ):

            nonempty_target_i = nonempty_target_i[mask_i].float()
            nonempty_probs_i = nonempty_probs_i[mask_i]
            empty_probs_i = empty_probs_i[mask_i]

            intersection = (nonempty_target_i * nonempty_probs_i).sum()
            precision = intersection / nonempty_probs_i.sum()
            recall = intersection / nonempty_target_i.sum()
            spec = ((1 - nonempty_target_i) * (empty_probs_i)).sum() / (1 - nonempty_target_i).sum()

            batch_precision.append(precision)
            batch_recall.append(recall)
            batch_spec.append(spec)

        precision = torch.stack(batch_precision)
        recall = torch.stack(batch_recall)
        spec = torch.stack(batch_spec)

        return (
            F.binary_cross_entropy_with_logits(
                precision, torch.ones_like(precision)
            ) + F.binary_cross_entropy_with_logits(
                recall, torch.ones_like(recall)
            ) + F.binary_cross_entropy_with_logits(spec, torch.ones_like(spec))
        )

    def _sem_scal_loss(self, pred, ssc_target) -> torch.Tensor:
        """Compute the Semantic Completion loss.

        Compute the per-class occupancy loss and average the result.
        """
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)
        mask = ssc_target != 255
        n_classes = pred.shape[1]

        batch_loss = []
        for pred_i, target_i, mask_i in zip(pred, ssc_target, mask):

            loss = 0
            count = 0

            for i in range(0, n_classes):

                # Get probability of class i
                p = pred_i[i, :, :, :]

                # Remove unknown voxels
                target_ori = target_i
                p = p[mask_i]
                target = target_i[mask_i]

                completion_target = torch.ones_like(target)
                completion_target[target != i] = 0
                completion_target_ori = torch.ones_like(target_ori).float()
                completion_target_ori[target_ori != i] = 0
                if torch.sum(completion_target) > 0:
                    count += 1.0
                    nominator = torch.sum(p * completion_target)
                    loss_class = 0
                    if torch.sum(p) > 0:
                        precision = nominator / (torch.sum(p))
                        loss_precision = F.binary_cross_entropy_with_logits(
                            precision, torch.ones_like(precision)
                        )
                        loss_class += loss_precision
                    if torch.sum(completion_target) > 0:
                        recall = nominator / (torch.sum(completion_target))
                        loss_recall = F.binary_cross_entropy_with_logits(
                            recall, torch.ones_like(recall)
                        )
                        loss_class += loss_recall
                    if torch.sum(1 - completion_target) > 0:
                        specificity = (
                            torch.sum((1 - p) * (1 - completion_target))
                            / torch.sum(1 - completion_target)
                        )
                        loss_specificity = F.binary_cross_entropy_with_logits(
                            specificity, torch.ones_like(specificity)
                        )
                        loss_class += loss_specificity
                    loss += loss_class

            batch_loss.append(loss / count)

        return torch.stack(batch_loss).mean()

    def forward(self, pred, ssc_target) -> torch.Tensor:
        sem_loss = self._sem_scal_loss(pred, ssc_target)
        geo_loss = self._geo_scal_loss(pred, ssc_target)
        return self.loss_weight * (self.alpha * sem_loss + self.beta * geo_loss)
