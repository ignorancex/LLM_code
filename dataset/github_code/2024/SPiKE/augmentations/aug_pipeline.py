"""
Module for tensor augmentation pipeline.
"""

import numpy as np
import torch

from augmentations.augmentation import Augmentation
from augmentations.rotation_aug import RotationAug
from augmentations.mirror_aug import MirrorAug
from augmentations.center_aug import CenterAug


class AugPipeline:
    """Class to represent an augmentation pipeline."""

    def __init__(self):
        """Constructor."""
        self.aug_classes_ = {sub.__name__: sub for sub in Augmentation.__subclasses__()}
        self.pipeline_ = []
        self.augmented_clip = []
        self.augmentation_params_list = []
        self.probabilities = []

    def create_pipeline(self, p_dict_list):
        """Create the pipeline.

        Args:
            p_dict_list (list of dict): List of dictionaries with the
                different augmentations in the pipeline.
        """
        self.pipeline_ = [
            self.aug_classes_[aug_dict["name"]](**aug_dict) for aug_dict in p_dict_list
        ]
        self.probabilities = [torch.rand(1).item() for _ in self.pipeline_]

    def augment(self, clip, gt, target_frame="last"):
        """
        Augment a sequence of tensors.

        Args:
            clip (torch.Tensor): Input point cloud sequence with dimensions [frames_per_clip, n_points, 3].
            gt (torch.Tensor): Ground truth joints with dimensions [1, n_joints, 3].
            target_frame (str): Ground truth position frame ('last' or 'middle').

        Returns:
            aug_clip (torch.Tensor): Augmented sequence with the same dimensions as the input [frames_per_clip, n_points, 3].
            params (list of tuples): List of parameters selected for each augmentation.
            aug_gt (torch.Tensor): Augmented ground truth with dimensions [n_joints, 3].
        """

        aug_clip = []
        aug_param_list = []
        prob = []

        mirror_probs = np.random.random(3)
        angle = torch.rand(1).item()

        if target_frame == "last":
            gt_frame_idx = len(clip) - 1
        elif target_frame == "middle":
            gt_frame_idx = len(clip) // 2
        else:
            raise ValueError("Invalid target_frame value. Use 'last' or 'middle'.")

        # Set augmentation parameters for the whole clip
        for cur_aug in enumerate(self.pipeline_):
            prob.append(torch.rand(1).item())
            _, cur_aug_object = cur_aug
            if isinstance(cur_aug_object, MirrorAug):
                cur_aug_object.set_probs(mirror_probs)
            elif isinstance(cur_aug_object, CenterAug):
                cur_aug_object.set_centroid(torch.mean(clip.view(-1, 3), dim=0))
            elif isinstance(cur_aug_object, RotationAug):
                cur_aug_object.set_angle(angle)

        aug_gt_tensor = (
            torch.from_numpy(gt[0]).to(torch.float32)
            if isinstance(gt[0], np.ndarray)
            else gt[0].to(torch.float32)
        )

        for i, p_tensor in enumerate(clip):
            if isinstance(p_tensor, np.ndarray):
                cur_tensor = torch.from_numpy(p_tensor).to(torch.float32)
            else:
                cur_tensor = p_tensor.to(torch.float32)
            cur_aug_param_list = []

            for j, cur_aug in enumerate(self.pipeline_):
                if prob[j] <= cur_aug.prob_:
                    apply_on_gt = (
                        i == gt_frame_idx
                    )  # Only apply augmentation on gt for the target frame
                    cur_tensor, cur_params, aug_gt_tensor = (
                        cur_aug.__compute_augmentation__(
                            cur_tensor, aug_gt_tensor, apply_on_gt
                        )
                    )

                    cur_aug_param_list.append((cur_aug.__class__.__name__, cur_params))
            aug_clip.append(cur_tensor.numpy())
            aug_param_list.append(cur_aug_param_list)

        return torch.FloatTensor(aug_clip), aug_param_list, aug_gt_tensor
