"""
Module for MirrorAug augmentation.
"""

import numpy as np
import torch
from augmentations.augmentation import Augmentation
from const.skeleton_joints import flip_joint_sides

class MirrorAug(Augmentation):
    """Class to represent a point cloud augmentation."""

    def __init__(self, p_prob=1.0, p_axes=None, **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_axes (list of bool): Axes to apply the augmentation.
        """
        if p_axes is None:
            p_axes = [True, True, False]
        self.axes_ = p_axes
        super().__init__(p_prob)

    def set_probs(self, probs):
        """Set the probabilities for the augmentation.

        Args:
            probs (numpy.ndarray): Probabilities for each axis.
        """
        self.probs = probs

    def __compute_augmentation__(self, p_pts, p_gt_tensor=None, apply_on_gt=True):
        """Implement the augmentation.

        Args:
            p_pts (tensor): Input tensor.
            p_gt_tensor (tensor, optional): Ground truth tensor. Defaults to None.
            apply_on_gt (bool, optional): Whether to apply augmentation on ground truth. Defaults to True.

        Returns:
            tuple: Augmented tensor, parameters selected for the augmentation, and ground truth tensor.
        """
        device = p_pts.device

        mask_1 = torch.tensor(self.probs, device=device) < self.prob_
        mask_2 = torch.tensor(self.axes_, device=device)
        mask = torch.logical_and(mask_1, mask_2)
        mirror_vec = (
            torch.ones(3, device=device) * (1.0 - mask.float())
            - torch.ones(3, device=device) * mask
        )

        aug_pts = p_pts * mirror_vec.view(1, -1)

        if apply_on_gt:
            # Flip coordinates
            augmented_gt_tensor = p_gt_tensor * mirror_vec.view(1, -1)
            # If flipping in X or Z dimension, exchange positions of left and right joints
            # Flipping in both X and Z dim cancels out
            flip = torch.logical_xor(mirror_vec[0] == -1, mirror_vec[2] == -1).item()
            if flip:
                augmented_gt_tensor = flip_joint_sides(augmented_gt_tensor)
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (mirror_vec,), augmented_gt_tensor
    