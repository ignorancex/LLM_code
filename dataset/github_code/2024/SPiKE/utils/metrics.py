"""
Module for utility functions used in training.
"""

import torch


def joint_accuracy(predicted, target, threshold):
    """
    Calculate the joint accuracy between predicted and target points.

    Args:
        predicted (torch.Tensor): Predicted joint positions.
        target (torch.Tensor): Ground truth joint positions.
        threshold (float): Distance threshold for considering a joint as correctly predicted.

    Returns:
        tuple: Percentage of correct keypoints (PCK) and mean average precision (mAP).
    """
    # Calculate Euclidean distance between predicted and target
    distance = torch.norm(predicted - target, dim=-1)

    correct = distance < threshold
    joint_wise_correct_points = correct.sum(dim=1).sum(dim=0)
    frame_wise_correct_points = correct.sum(dim=-1)

    n_frames = target.shape[0] * target.shape[1]
    n_joints = target.shape[2]

    pck = (joint_wise_correct_points / n_frames) * 100
    mean_ap = torch.mean(frame_wise_correct_points.float() / n_joints) * 100

    return pck, mean_ap
