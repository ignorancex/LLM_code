"""
Module for CenterAug augmentation.
"""

import numpy as np
import torch
from augmentations.augmentation import Augmentation


class CenterAug(Augmentation):
    """Class to represent a point cloud augmentation."""

    def __init__(self, p_prob=1.0, p_axes=None, **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_axes (list of bool): Axes to apply the augmentation.
        """
        if p_axes is None:
            p_axes = [True, True, True]
        self.axes_ = p_axes
        self.centroid = None

        # Super class init.
        super().__init__(p_prob)

    def set_centroid(self, centroid):
        """Set the centroid for the augmentation.

        Args:
            centroid (tensor): The centroid to set.
        """
        self.centroid = centroid

    def __compute_augmentation__(self, p_pts, p_gt_tensor=None, apply_on_gt=True):
        """Implement the augmentation.

        Args:
            p_pts (tensor): Input tensor.
            p_gt_tensor (tensor, optional): Ground truth tensor. Defaults to None.
            apply_on_gt (bool, optional): Whether to apply augmentation on ground truth. Defaults to True.

        Returns:
            tuple: Augmented tensor, parameters selected for the augmentation, and ground truth tensor.
        """
        axes_mask = np.logical_not(np.array(self.axes_))
        center_pt = self.centroid if self.centroid is not None else torch.mean(p_pts, 0)
        aug_pts = p_pts - center_pt.reshape((1, -1))
        aug_pts[:, axes_mask] = p_pts[:, axes_mask]

        if apply_on_gt:
            augmented_gt_tensor = p_gt_tensor - center_pt.reshape((1, -1))
            augmented_gt_tensor[:, axes_mask] = p_gt_tensor[:, axes_mask]
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (center_pt, axes_mask), augmented_gt_tensor
