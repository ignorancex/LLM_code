import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


def generate_density_map(scene_size, xy_coords, mode='replace', clip_value=3, blur_kernel_size=75, blur_sigma=15, normalize=False):
    """
    Generate a density map from given coordinates.
    
    Params:
        scene_size (tuple): Size of the scene as (width, height).
        xy_coords (np.ndarray): Coordinates of points to generate the density map, shape (N, 2).
        mode (str): Mode of density map generation, either 'add' or 'replace'.
        clip_value (int): Maximum value for clipping the density map.
        blur_kernel_size (int): Size of the Gaussian kernel for blurring.
        blur_sigma (float): Standard deviation for Gaussian kernel.
        normalize (bool): Whether to normalize the density map.
    
    Returns:
        density_map (np.ndarray): Generated density map, shape (height, width).
    """

    scene_width, scene_height = scene_size
    xy_coords = np.round(xy_coords).astype(np.int32)

    if mode == 'add':
        density_map = np.zeros((scene_height, scene_width), dtype=np.uint32)
        np.add.at(density_map, (xy_coords[:, 1], xy_coords[:, 0]), 1)
        density_map = np.clip(density_map, 0, clip_value)
        density_map = density_map.astype(np.float32) / clip_value * 255
        density_map = density_map.astype(np.uint8)
    elif mode == 'replace':
        density_map = np.zeros((scene_height, scene_width), dtype=np.uint8)
        density_map[xy_coords[:, 1], xy_coords[:, 0]] = 255
    else:
        raise ValueError(f'Invalid mode: {mode}')
    
    density_map = gaussian_filter(density_map.astype(np.float32), sigma=blur_sigma, mode='reflect')
    density_map = np.log1p(density_map) if normalize else density_map
    density_map = density_map.astype(np.float32) / density_map.max()
    return density_map


def generate_flow_map(scene_size, trajectories, blur_sigma=15):
    """
    Generate a flow map from given trajectories.
    
    Params:
        scene_size (tuple): Size of the scene as (width, height).
        trajectories (list): List of trajectories, each trajectory is a list of (x, y) coordinates.
        blur_sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        flow_map (np.ndarray): Generated flow map, shape (height, width, 2).
    """

    scene_width, scene_height = scene_size
    flow_map = np.zeros((scene_height, scene_width, 2), dtype=np.float32)
    count_map = np.zeros((scene_height, scene_width), dtype=np.int32)

    for trajectory in trajectories:
        traj = np.array(trajectory)[:, -2:]
        displacements = traj[1:] - traj[:-1]
        displacements /= np.linalg.norm(displacements, axis=1)[:, None] + 1e-6
        displacements = np.abs(displacements)

        positions = np.round(traj[:-1]).astype(int)
        ix = np.clip(positions[:, 0], 0, scene_width - 1)
        iy = np.clip(positions[:, 1], 0, scene_height - 1)

        np.add.at(flow_map[:, :, 0], (iy, ix), displacements[:, 0])
        np.add.at(flow_map[:, :, 1], (iy, ix), displacements[:, 1])
        np.add.at(count_map, (iy, ix), 1)

    # Average the flow vectors
    nonzero_indices = count_map > 0
    flow_map[nonzero_indices] /= count_map[nonzero_indices, None]

    # Interpolate missing values
    known_points = np.argwhere(nonzero_indices)
    unknown_points = np.argwhere(~nonzero_indices)
    if len(known_points) > 0 and len(unknown_points) > 0:
        known_values_x = flow_map[known_points[:, 0], known_points[:, 1], 0]
        known_values_y = flow_map[known_points[:, 0], known_points[:, 1], 1]
        interp_x = griddata(known_points, known_values_x, unknown_points, method='nearest')
        interp_y = griddata(known_points, known_values_y, unknown_points, method='nearest')
        flow_map[unknown_points[:, 0], unknown_points[:, 1], 0] = interp_x
        flow_map[unknown_points[:, 0], unknown_points[:, 1], 1] = interp_y
    
    # Gaussian smoothing
    flow_map[:, :, 0] = gaussian_filter(flow_map[:, :, 0], sigma=blur_sigma)
    flow_map[:, :, 1] = gaussian_filter(flow_map[:, :, 1], sigma=blur_sigma)
    flow_map = np.clip(flow_map, 0, 1)
    return flow_map


def sampling_xy_pos(probability_map, num_samples, rel_threshold=None, replacement=False):
    """
    Sample (x, y) coordinates from a 2D probability map.

    Params:
        probability_map (torch.Tensor): Probability map of shape [batch, H, W].
        num_samples (int): Number of samples to draw.
        rel_threshold (float, optional): Relative threshold for sampling.
        replacement (bool): Whether to sample with replacement.
    Returns:
        preds (torch.Tensor): Sampled coordinates of shape [batch, num_samples, 2].
    """

    B, H, W = probability_map.shape
    prob_map = probability_map.view(B, H * W)  # shape=[batch, H*W]
    if rel_threshold is not None:
        thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(B, H * W)
        mask = prob_map < thresh_values * rel_threshold
        prob_map = prob_map * (~mask).int()
        prob_map = prob_map / prob_map.sum()

    # Sample indices from the probability map
    samples = torch.multinomial(prob_map, num_samples=num_samples, replacement=replacement)
    samples = samples.view(B, num_samples).float()

    # Unravel sampled idx into coordinates of shape [batch, sample, 2]
    preds = torch.zeros(B, num_samples, 2, device=probability_map.device)
    preds[..., 0] = samples % W
    preds[..., 1] = torch.floor(samples / W)

    return preds
