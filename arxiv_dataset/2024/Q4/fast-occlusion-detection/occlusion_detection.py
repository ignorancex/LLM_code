import torch
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt

edge_threshold = 1.0
distance_threshold = 2.0
disparity_threshold = 0.5
ADD_MASK_PIXELS = 6
NEIGHBORS = 4

def get_baseline(x0, y0, x1, y1):
    baseline = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    return baseline

def get_angle(x0, y0, x1, y1):
    angle = math.atan2(y1 - y0, x1 - x0)
    return angle

def run(ms_image_raw, disparity, calibration_matrices, camera_pos_array):
    center_cam_idx = ms_image_raw.shape[1]//2
    baselines = []
    angles = []
    min_baseline = np.iinfo(int).max
    center_pos = camera_pos_array[center_cam_idx]
    for i in range(ms_image_raw.shape[1]):
        current_pos = camera_pos_array[i]
        baseline = get_baseline(current_pos[0], current_pos[1], center_pos[0], center_pos[1])
        angle = get_angle(current_pos[0], current_pos[1], center_pos[0], center_pos[1])
        if baseline == 0:
            baseline = np.nan

        baselines.append(baseline)
        angles.append(angle)

        if baseline < min_baseline:
            min_baseline = baseline
    #Global stuff for mask
    kernel = disparity.new_ones((2, 1))
    kernel[0, 0] = -1
    kernel = kernel[None, None, ...]

    depth_gradient_y = F.conv2d(F.pad(disparity[:, 0], (0, 0, 0, 1), mode='replicate'), kernel, padding=0)
    depth_gradient_x = F.conv2d(F.pad(disparity[:, 0], (0, 1, 0, 0), mode='replicate'), torch.transpose(kernel, 2, 3), padding=0)

    depth_gradient_mag = torch.sqrt(depth_gradient_x ** 2 + depth_gradient_y ** 2)
    depth_gradient_angle = torch.atan2(depth_gradient_y, depth_gradient_x)

    ms_image_masked = torch.zeros_like(ms_image_raw)
    ms_image_masked[:, center_cam_idx] = ms_image_raw[:, center_cam_idx]
    ms_mask = torch.zeros_like(ms_image_masked)
    ms_mask[:, center_cam_idx, :, :] = 1

    for i in range(ms_image_raw.shape[1]):
        if i == center_cam_idx:
            continue

        baseline_ratio = baselines[i]/min_baseline
        image, mask = warp_cam(ms_image_raw[:, i].unsqueeze(dim=1), disparity * baseline_ratio, depth_gradient_mag * baseline_ratio, depth_gradient_angle, calibration_matrices[i], -angles[i])
        ms_image_masked[:, i] = image[:, 0]
        ms_mask[:, i] = mask[:, 0]

    return ms_image_masked * ms_mask, ms_mask

def warp_cam(img, disp, depth_gradient_mag, depth_gradient_angle, calibration_matrix, angle):
    dX = math.cos(angle)
    dY = math.sin(angle)

    torch_horizontal = torch.arange(img.size(3), device=img.device, dtype=torch.float).view(1, 1, 1, img.size(3)).expand(img.size(0), 1, img.size(2), img.size(3))
    torch_vertical = torch.arange(img.size(2), device=img.device, dtype=torch.float).view(1, 1, img.size(2), 1).expand(img.size(0), 1, img.size(2), img.size(3))

    inv_calib = torch.linalg.inv(calibration_matrix)
    denom = inv_calib[2, 0] * torch_horizontal + inv_calib[2, 1] * torch_vertical + inv_calib[2, 2]
    torch_horizontal = (inv_calib[0, 0] * torch_horizontal + inv_calib[0, 1] * torch_vertical + inv_calib[0, 2]) / denom + dX * disp
    torch_vertical = (inv_calib[1, 0] * torch_horizontal + inv_calib[1, 1] * torch_vertical + inv_calib[1, 2]) / denom + dY * disp
    mask = (torch_horizontal > -0.5) * (torch_horizontal < img.size(3) - 0.5) * (torch_vertical > -0.5) * (torch_vertical < img.size(2) - 0.5)
    tensor_grid = torch.cat([2 * torch_horizontal/(img.size(3) - 1) - 1, 2 * torch_vertical/(img.size(2) - 1) - 1], dim=1)

    img_warped = torch.nn.functional.grid_sample(input=img, grid=tensor_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)

    mask[mask == 1] = get_mask(disp, depth_gradient_mag, depth_gradient_angle, angle)[mask == 1]

    B, C, H, W = mask.shape
    # Solve problem of sharp depth map vs unsharp image -> glow/bloom
    for shift in range(1, ADD_MASK_PIXELS + 1):
        shift_y = round(dY * shift)
        shift_x = round(dX * shift)
        shift_start_y = max(shift_y, 0)
        shift_start_x = max(shift_x, 0)
        shift_end_y = min(shift_y + mask.shape[2], mask.shape[2])
        shift_end_x = min(shift_x + mask.shape[3], mask.shape[3])
        orig_start_y = max(-shift_y, 0)
        orig_start_x = max(-shift_x, 0)
        orig_end_y = min(-shift_y + mask.shape[2], mask.shape[2])
        orig_end_x = min(-shift_x + mask.shape[3], mask.shape[3])
        shifted_mask = mask.new_zeros([B, C, H, W])
        shifted_mask[:, :, shift_start_y:shift_end_y, shift_start_x:shift_end_x] = mask[:, :, orig_start_y:orig_end_y, orig_start_x:orig_end_x]
        mask[mask == 1] = shifted_mask[mask == 1]

    return img_warped, mask

def get_mask(depth_map, depth_gradient_mag, depth_gradient_angle, angle):
    max_disp_diff = torch.max(depth_map) - torch.min(depth_map)
    line_search_size = int(torch.ceil(max_disp_diff * 2))
    if line_search_size % 2 == 0:
        line_search_size += 1

    #Calculate relevant edges for finding pixels to mask
    edges = depth_gradient_mag >= edge_threshold

    angle_difference = (depth_gradient_angle[edges] - torch.pi) - angle
    angle_difference[angle_difference > torch.pi] -= 2 * torch.pi
    angle_difference[angle_difference < -torch.pi] += 2 * torch.pi
    edges[edges > 0] = torch.abs(angle_difference) < torch.pi/2

    _, coords_y, coords_x = torch.where(edges > 0)
    line_coords = torch.arange(line_search_size, device=coords_x.device) - line_search_size//2
    line_coords = torch.tile(line_coords[None, :], (coords_x.shape[0], 1))
    scan_x = (math.cos(angle) * line_coords + coords_x[:, None]).long()
    scan_y = (math.sin(angle) * line_coords + coords_y[:, None]).long()
    valid_indices = (scan_y >= 0) * (scan_x >= 0) * (scan_y < depth_map.shape[2]) * (scan_x < depth_map.shape[3])
    invalid_indices = ~valid_indices
    #Set invalid indices to 0
    scan_x[invalid_indices] = 0
    scan_y[invalid_indices] = 0

    scan_depth_lines = depth_map[0, 0, scan_y, scan_x]
    scan_depth_lines[invalid_indices] = torch.min(depth_map) #Invalid indices should never occlude something

    local_warped = line_coords + scan_depth_lines
    sorted_ind = torch.argsort(local_warped, dim=1)
    sorted_local = torch.take_along_dim(local_warped, sorted_ind, dim=1)
    sorted_depth = torch.take_along_dim(scan_depth_lines, sorted_ind, dim=1)
    sorted_scan_x = torch.take_along_dim(scan_x, sorted_ind, dim=1)
    sorted_scan_y = torch.take_along_dim(scan_y, sorted_ind, dim=1)
    sorted_invalid_indices = torch.take_along_dim(invalid_indices, sorted_ind, dim=1)

    local_mask = torch.zeros_like(sorted_local)
    for i in range(1, NEIGHBORS + 1):
        diff = sorted_local[:, i:] - sorted_local[:, :-i]
        diff_depth = sorted_depth[:, i:] - sorted_depth[:, :-i]

        local_mask[:, :-i] += (diff < distance_threshold) * (diff_depth > disparity_threshold)
        local_mask[:, i:] += (diff < distance_threshold) * (diff_depth < -disparity_threshold)

    local_mask[local_mask > 0] = 1
    local_mask[sorted_invalid_indices] = 0
    bad_x = sorted_scan_x[local_mask == 1]
    bad_y = sorted_scan_y[local_mask == 1]

    mask = torch.ones_like(depth_map, dtype=torch.bool)
    mask[:, :, bad_y, bad_x] = 0

    return mask
