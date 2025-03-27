"""
Module for RotationAug augmentation.
"""

import numpy as np
import torch
from augmentations.augmentation import Augmentation


class RotationAug(Augmentation):
    """Class to represent a point cloud augmentation."""

    def __init__(
        self,
        p_prob=1.0,
        p_axis=0,
        p_min_angle=0,
        p_max_angle=2 * np.pi,
        p_angle_values=None,
        **kwargs
    ):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_axis (int): Index of the axis.
            p_min_angle (float): Minimum rotation angle.
            p_max_angle (float): Maximum rotation angle.
            p_angle_values (list of floats): User-defined angle per epoch.
        """
        self.axis_ = p_axis
        self.min_angle_ = p_min_angle
        self.max_angle_ = p_max_angle
        self.angle_values_ = p_angle_values
        self.cur_angle = None

        # Super class init.
        super().__init__(p_prob)

    def set_angle(self, angle):
        """Set the angle for the augmentation.

        Args:
            angle (float): Angle to set.
        """
        if self.angle_values_ is None:
            self.cur_angle = (
                angle * (self.max_angle_ - self.min_angle_) + self.min_angle_
            )
        else:
            self.cur_angle = self.angle_values_[self.epoch_iter_]

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
        if self.axis_ == 0:
            rotation_matrix = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(self.cur_angle), -np.sin(self.cur_angle)],
                    [0.0, np.sin(self.cur_angle), np.cos(self.cur_angle)],
                ],
                device=device,
                dtype=torch.float32,
            )
        elif self.axis_ == 1:
            rotation_matrix = torch.tensor(
                [
                    [np.cos(self.cur_angle), 0.0, np.sin(self.cur_angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(self.cur_angle), 0.0, np.cos(self.cur_angle)],
                ],
                device=device,
                dtype=torch.float32,
            )
        elif self.axis_ == 2:
            rotation_matrix = torch.tensor(
                [
                    [np.cos(self.cur_angle), -np.sin(self.cur_angle), 0.0],
                    [np.sin(self.cur_angle), np.cos(self.cur_angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )

        aug_pts = torch.matmul(p_pts, rotation_matrix)

        if apply_on_gt:
            augmented_gt_tensor = torch.matmul(p_gt_tensor, rotation_matrix)
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (self.axis_, self.cur_angle), augmented_gt_tensor
    