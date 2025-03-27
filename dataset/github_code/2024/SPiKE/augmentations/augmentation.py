"""
Module for tensor augmentation.
"""

from abc import ABC, abstractmethod

class Augmentation(ABC):
    """Class to represent a tensor augmentation."""

    def __init__(self, p_prob):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
        """
        self.prob_ = p_prob
        self.epoch_iter_ = 0

    def increase_epoch_counter(self):
        """Method to update the epoch counter for user-defined augmentations."""
        self.epoch_iter_ += 1

    @abstractmethod
    def __compute_augmentation__(self, p_tensor, gt_tensor=None, apply_on_gt=True):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            gt_tensor (tensor, optional): Ground truth tensor. Defaults to None.
            apply_on_gt (bool, optional): Whether to apply augmentation on ground truth. Defaults to True.

        Returns:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            gt_tensor (tensor): (Augmented) ground truth tensor.
        """
        pass
    