import os
import numpy as np
import imageio
from glob import glob
import cv2 as cv
import scipy.ndimage


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)


def enhanced_confidence_weighted_depth_aggregation_with_variations(depth_paths, depth_scale, confidence_threshold=0.3,
                                                                variation_weight=0.5):
    """
    Enhanced depth aggregation with hole filling and geometric consistency.
    """
    depth_frames = []
    confidence_frames = []
    variation_frames = []

    for depth_path in depth_paths:
        # Read and scale depth image
        depth_image = imread(depth_path).astype(np.float32) / depth_scale
        if depth_image.ndim == 3:
            if depth_image.shape[2] == 4:
                depth_image = depth_image[:, :, :3]
            depth_image = np.dot(depth_image[..., :3], [0.2989, 0.5870, 0.1140])

        # Multi-scale bilateral filtering
        depth_filtered = depth_image.copy()
        for sigma in [3, 5, 7]:  # Multiple scales for better hole filling
            depth_temp = cv.bilateralFilter(depth_filtered, d=sigma*2+1, sigmaColor=0.1, sigmaSpace=sigma)
            holes = (depth_filtered == 0)
            depth_filtered[holes] = depth_temp[holes]

        depth_frames.append(depth_filtered)

        # Enhanced multi-scale confidence
        near = np.percentile(depth_filtered[depth_filtered > 0], 2)
        far = np.percentile(depth_filtered[depth_filtered > 0], 99)
        
        # Range confidence with relaxed boundaries
        range_conf = np.where((depth_filtered > near*0.9) & (depth_filtered < far*1.1), 1, 0).astype(np.float32)
        
        # Multi-scale edge confidence
        edge_conf = np.ones_like(depth_filtered)
        for scale in [1, 2, 4]:
            grad_x = cv.Sobel(depth_filtered, cv.CV_32F, 1, 0, ksize=2*scale+1)
            grad_y = cv.Sobel(depth_filtered, cv.CV_32F, 0, 1, ksize=2*scale+1)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edge_conf *= np.exp(-edge_magnitude / (edge_magnitude.mean() + 1e-8))

        # Local structure confidence
        structure_tensor = cv.cornerEigenValsAndVecs(depth_filtered, 3, 3)
        coherence = structure_tensor[:,:,0] / (structure_tensor[:,:,1] + 1e-8)
        structure_conf = np.exp(-coherence / (coherence.mean() + 1e-8))
        
        # Combine confidences
        confidence = range_conf * edge_conf * structure_conf
        confidence_frames.append(np.clip(confidence, 0, confidence_threshold))

        # Improved variation computation with structure preservation
        variation_map = np.zeros_like(depth_filtered)
        for scale in [1, 2, 4]:
            laplacian = cv.Laplacian(depth_filtered, cv.CV_32F, ksize=2*scale+1)
            variation_map += np.abs(laplacian) * np.exp(-scale)
        
        variation_map = cv.GaussianBlur(variation_map, (5, 5), 0)
        variation_frames.append(variation_map)

    depth_frames = np.array(depth_frames)
    confidence_frames = np.array(confidence_frames)
    variation_frames = np.array(variation_frames)

    # Weighted depth with temporal consistency
    temporal_weight = np.exp(-np.abs(np.diff(depth_frames, axis=0)))
    temporal_weight = np.pad(temporal_weight, ((0,1), (0,0), (0,0)), mode='edge')
    
    weighted_depth_sum = np.sum(depth_frames * confidence_frames * temporal_weight, axis=0)
    confidence_sum = np.sum(confidence_frames * temporal_weight, axis=0)
    
    aggregated_depth = np.divide(weighted_depth_sum, confidence_sum + 1e-8,
                               out=np.zeros_like(weighted_depth_sum),
                               where=confidence_sum > 0)

    # Fill remaining holes using multi-scale diffusion
    holes = (aggregated_depth == 0)
    if holes.any():
        filled_depth = aggregated_depth.copy()
        for scale in [3, 7, 15]:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (scale, scale))
            dilated = cv.dilate(filled_depth, kernel)
            filled_depth[holes] = dilated[holes]
        
        # Smooth transition between filled and original values
        transition_mask = cv.dilate((holes).astype(np.float32), kernel) - holes.astype(np.float32)
        aggregated_depth = (1 - transition_mask) * aggregated_depth + transition_mask * filled_depth

    # Adaptive variation incorporation
    mean_variation = np.mean(variation_frames, axis=0)
    normalized_variation = mean_variation / (mean_variation.max() + 1e-8)
    
    # Structure-preserving variation blending
    structure_weight = structure_conf * variation_weight
    enhanced_depth = (1 - structure_weight) * aggregated_depth + structure_weight * normalized_variation

    return enhanced_depth

def enhanced_confidence_weighted_depth_aggregation(depth_paths, depth_scale, confidence_threshold=0.5):
    """
    Enhanced depth aggregation with improved confidence weighting for surgical scenes.
    """
    depth_frames = []
    confidence_frames = []

    for depth_path in depth_paths:
        # Read and scale depth image
        depth_image = imread(depth_path).astype(np.float32) / depth_scale

        if depth_image.ndim == 3:
            if depth_image.shape[2] == 4:
                depth_image = depth_image[:, :, :3]
            depth_image = np.dot(depth_image[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Bilateral filtering for noise reduction
        depth_filtered = cv.bilateralFilter(depth_image.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=5)
        depth_frames.append(depth_filtered)

        # Enhanced confidence computation
        near = np.percentile(depth_filtered, 3)
        far = np.percentile(depth_filtered, 98)
        
        # Depth range confidence
        range_conf = np.where((depth_filtered > near) & (depth_filtered < far), 1, 0).astype(np.float32)
        
        # Edge confidence
        grad_x = cv.Sobel(depth_filtered, cv.CV_32F, 1, 0, ksize=3)
        grad_y = cv.Sobel(depth_filtered, cv.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_conf = np.exp(-edge_magnitude / edge_magnitude.mean())
        
        # Local smoothness confidence
        smoothness = 1.0 - cv.GaussianBlur(edge_magnitude, (5, 5), 0) / (edge_magnitude.max() + 1e-8)
        
        # Combine confidences
        confidence = range_conf * edge_conf * smoothness
        confidence_frames.append(np.clip(confidence, 0, confidence_threshold))

    depth_frames = np.array(depth_frames)
    confidence_frames = np.array(confidence_frames)

    # Weighted aggregation
    weighted_depth_sum = np.sum(depth_frames * confidence_frames, axis=0)
    confidence_sum = np.sum(confidence_frames, axis=0)
    
    # Handle division by zero
    aggregated_depth = np.divide(weighted_depth_sum, confidence_sum + 1e-8, 
                               out=np.zeros_like(weighted_depth_sum),
                               where=confidence_sum > 0)

    return aggregated_depth


def get_all_initial_data_endo(path, data_type, depth_scale, is_mask, npy_file):
    poses_bounds = np.load(os.path.join(path, npy_file))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]
    H = H.astype(int)
    W = W.astype(int)

    depth_paths = sorted(glob(os.path.join(path, 'depth/*')))
    color_paths = sorted(glob(os.path.join(path, 'images/*')))
    masks_path = sorted(glob(os.path.join(path, 'gt_masks/*')))

    # Generate invisible and dilated invisible masks
    inpaint_mask_all = np.zeros((H, W))
    if is_mask:
        for mask_path in masks_path:
            mask_image = 1.0 - (np.array(imread(mask_path)) / 255.0)
            inpaint_mask_all += mask_image
            inpaint_mask_all[inpaint_mask_all >= 1] = 1

        inpaint_mask_all = (1.0 - inpaint_mask_all) * 255.0
        inpaint_mask_all = inpaint_mask_all.astype(np.uint8)
        imageio.imwrite(os.path.join(path, "invisible_mask.png"), inpaint_mask_all)
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv.dilate(inpaint_mask_all, kernel, iterations=2)
        if data_type == 'endonerf':
            dilated_mask[-12:, :] = 255
        imageio.imwrite(os.path.join(path, "dilated_invisible_mask.png"), dilated_mask)

    # Aggregate depth maps with confidence weighting and variations
    depth_all = enhanced_confidence_weighted_depth_aggregation(depth_paths, depth_scale)

    # Aggregate color maps with confidence weighting
    color_frames = []
    confidence_frames = []

    for depth_path, color_path in zip(depth_paths, color_paths):
        # Read confidence from depth processing
        depth_image = imread(depth_path).astype(np.float32) / depth_scale

        if depth_image.ndim == 3:
            if depth_image.shape[2] == 4:
                depth_image = depth_image[:, :, :3]
            depth_image = np.dot(depth_image[..., :3], [0.2989, 0.5870, 0.1140])

        near = np.percentile(depth_image, 3)
        far = np.percentile(depth_image, 98)
        confidence_mask = np.where((depth_image > near) & (depth_image < far), 1, 0).astype(np.float32)

        image = imread(color_path).astype(np.float32) / 255.0
        color_frames.append(image * confidence_mask[..., np.newaxis])
        confidence_frames.append(confidence_mask)

    color_frames = np.array(color_frames)
    confidence_frames = np.array(confidence_frames)

    aggregated_color = np.sum(color_frames, axis=0) / (np.sum(confidence_frames, axis=0)[..., np.newaxis] + 1e-8)

    # Aggregate mask information
    mask_all = np.zeros((H, W))
    for i in range(len(depth_paths)):
        mask_image = np.ones((H, W))
        if is_mask and i < len(masks_path):
            mask_image = np.array(imread(masks_path[i]) / 255.0)
            mask_image = 1.0 - np.where(mask_image > 0.5, 1.0, 0.0)

        mask_depth = (depth_all > 0) * mask_image
        mask_all += mask_depth
        mask_all[mask_all >= 1] = 1

    intrinsics = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])

    return aggregated_color, depth_all, intrinsics, mask_all


def get_pointcloud(color, depth, intrinsics, mask, w2c=None, transform_pts=False):
    # Ensure depth has the correct shape
    if len(depth.shape) == 2:  # If depth is (H, W)
        depth = depth[np.newaxis, ...]  # Add a batch dimension to make it (1, H, W)
    elif len(depth.shape) != 3 or depth.shape[0] != 1:
        raise ValueError(f"Unexpected depth shape: {depth.shape}. Expected (1, H, W).")

    height, width = depth.shape[1:]  # Get height and width from depth
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = np.meshgrid(np.arange(width).astype(np.float32),
                                 np.arange(height).astype(np.float32),
                                 indexing='xy')
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY

    # Reshape to flat arrays for computation
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Ensure xx, yy, and depth_z have the same size
    if xx.size != depth_z.size or yy.size != depth_z.size:
        raise ValueError(f"Resolution mismatch: xx ({xx.size}), yy ({yy.size}), depth_z ({depth_z.size})")

    # Initialize point cloud
    pts_cam = np.stack((xx * depth_z, yy * depth_z, depth_z), axis=-1)

    if transform_pts:
        pix_ones = np.ones((depth_z.size, 1)).astype(np.float32)
        pts4 = np.concatenate((pts_cam, pix_ones), axis=1)
        c2w = np.linalg.inv(w2c)
        pts = np.dot(pts4, c2w.T)[:, :3]
    else:
        pts = pts_cam

    # Ensure the color array has the correct shape
    if color.shape == (height, width, 3):  # If (H, W, C), transpose to (C, H, W)
        color = np.transpose(color, (2, 0, 1))
    elif color.shape != (3, height, width):
        raise ValueError(f"Unexpected color shape: {color.shape}. Expected (3, {height}, {width}).")

    print(f"Color array min: {color.min()}, max: {color.max()}, dtype: {color.dtype}")

    cols = np.transpose(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    mask_sample = sample_pts(height, width, 3)
    mask_sample = (mask_sample != 0)
    mask_sample = mask_sample.reshape(-1)

    # Convert mask to boolean type
    mask = mask.astype(bool).reshape(-1)

    print(f"Mask sum: {mask.sum()}, shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask sample sum: {mask_sample.sum()}, shape: {mask_sample.shape}, dtype: {mask_sample.dtype}")
    print(f"Mask sample min: {mask_sample.min()}, max: {mask_sample.max()}, dtype: {mask_sample.dtype}")

    # Combine masks using bitwise AND
    mask_indices = np.where(mask & mask_sample)[0]
    pts = pts[mask_indices]
    cols = cols[mask_indices]

    print(f"Mask indices count: {len(mask_indices)}")

    print(f"Masked colors: min={cols.min()}, max={cols.max()}, shape={cols.shape}")

    print(f"Using the {pts.shape[0]} points as initial")
    return pts, cols * 255.0  # , mask_indices


def sample_pts(height, width, factor=2):
    mask_sample_h = np.zeros((height, width)).astype(int)
    mask_sample_w = np.zeros((height, width)).astype(int)
    mask_sample_h[:, 1::factor] = 1
    mask_sample_w[1::factor, :] = 1
    mask_sample = mask_sample_h & mask_sample_w

    return mask_sample
