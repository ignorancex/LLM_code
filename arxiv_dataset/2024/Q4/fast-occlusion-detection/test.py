import imageio.v3
import torch
import numpy as np
import os

import pfmio
import occlusion_detection

def load(dataset, index):
    string_index = str(index).zfill(7)
    occ = imageio.v3.imread(f"data/{string_index}.png")//255
    disp = -pfmio.readPFM(f"data/{string_index}.pfm").copy()

    occ = torch.from_numpy(occ)[None, None, ...]
    disp = torch.from_numpy(disp)[None, None, ...]

    occ = occ.bool()

    return disp, occ

def get_color_disparity(disparity):
    value = (disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity))
    value[value != value] = 0
    value = 1 - value

    green_increase = 185
    red_blue_exchange = 255
    green_decrease = 215
    total_travel = green_increase + red_blue_exchange + green_decrease
    green_increase_ratio = green_increase / total_travel
    red_blue_exchange_ratio = red_blue_exchange / total_travel
    green_decrease_ratio = green_decrease / total_travel

    rgb_image = np.zeros((value.shape[0], value.shape[1], 3))

    mask = value <= green_increase_ratio
    perc = value / green_increase_ratio
    rgb_image[mask, 0] = 255
    rgb_image[mask, 1] = 70 + perc[mask] * green_increase
    rgb_image[mask, 2] = 0

    mask = (value > green_increase_ratio) * (value <= green_increase_ratio + red_blue_exchange_ratio)
    perc = (value - green_increase_ratio) / red_blue_exchange_ratio
    rgb_image[mask, 0] = red_blue_exchange * (1 - perc[mask])
    rgb_image[mask, 1] = 255
    rgb_image[mask, 2] = red_blue_exchange * perc[mask]

    mask = value > green_increase_ratio + red_blue_exchange_ratio
    perc = (value - green_increase_ratio - red_blue_exchange_ratio) / green_decrease_ratio
    rgb_image[mask, 0] = 0
    rgb_image[mask, 1] = 255 - perc[mask] * green_decrease
    rgb_image[mask, 2] = 255

    rgb_image = rgb_image.astype(np.uint8)
    return rgb_image

def save_mask(path, mask):
    mask = mask[0, 0].detach().cpu().numpy()
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask == 1] = [255, 0, 0]

    imageio.imwrite(path, rgb, quality=90)

def save_prediction(path, pred, gt):
    pred = pred[0, 0].detach().cpu().numpy()
    gt = gt[0, 0].detach().cpu().numpy()
    rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    rgb[(pred == 0) * (gt == 0)] = [0, 0, 0]
    rgb[(pred == 0) * (gt == 1)] = [255, 255, 0]
    rgb[(pred == 1) * (gt == 0)] = [255, 0, 255]
    rgb[(pred == 1) * (gt == 1)] = [255, 0, 0]

    imageio.imwrite(path, rgb, quality=95)

def run(gpu=True):
    os.makedirs('out', exist_ok=True)

    disparity, occlusion_gt = load('val', 42)
    if gpu:
        disparity = disparity.cuda()
        occlusion_gt = occlusion_gt.cuda()
    ms_image_raw = disparity.new_ones((disparity.shape[0], 2, disparity.shape[2], disparity.shape[3]))

    rgb_disp = get_color_disparity(disparity[0, 0].detach().cpu().numpy())
    imageio.v3.imwrite(f'out/disp.jpg', rgb_disp, quality=90)
    save_mask(f'out/occ.jpg', occlusion_gt)

    # Pipeline
    calibration_matrices = [torch.eye(3)] * 2
    camera_pos_array = [[1, 0], [0, 0]]
    occlusion_detection.ADD_MASK_PIXELS = 0

    _, ms_mask = occlusion_detection.run(ms_image_raw, disparity, calibration_matrices, camera_pos_array)

    mask = ms_mask[:, 0] > 0.5
    mask = mask.unsqueeze(dim=1)
    mask = ~mask

    save_prediction(f'out/prediction.jpg', mask, occlusion_gt)

if __name__ == '__main__':
    run()
