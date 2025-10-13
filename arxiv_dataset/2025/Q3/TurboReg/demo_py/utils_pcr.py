import torch
import numpy as np

def numpy_to_torch32(device, *arrays):
    return [torch.tensor(array, device=device, dtype=torch.float32) for array in arrays]

def compute_transformation_error(trans_gt: np.ndarray, trans_pred: np.ndarray):
    """
    Compute the transformation error between the predicted and ground truth 4x4 rigid transformation matrices.

    This function calculates both the Relative Rotation Error (RRE) and Relative Translation Error (RTE).
    It extracts the rotation and translation components from the 4x4 matrices and computes the respective errors.

    Args:
        trans_gt (np.ndarray): Ground truth 4x4 transformation matrix
        trans_pred (np.ndarray): Predicted 4x4 transformation matrix

    Returns:
        rre (float): Relative Rotation Error in degrees
        rte (float): Relative Translation Error
    """
    
    # Extract rotation and translation components
    gt_rotation = trans_gt[:3, :3]
    est_rotation = trans_pred[:3, :3]
    gt_translation = trans_gt[:3, 3]
    est_translation = trans_pred[:3, 3]
    
    # Compute Relative Rotation Error (RRE)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    
    # Compute Relative Translation Error (RTE)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    
    return rre, rte

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)