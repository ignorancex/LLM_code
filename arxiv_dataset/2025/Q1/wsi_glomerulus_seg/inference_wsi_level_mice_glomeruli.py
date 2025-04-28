import os
from pathlib import Path

from mmseg.apis import init_model, inference_model

from tqdm import tqdm

import cv2
import torch
import numpy as np
import tifffile

import utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="path to WSI file")
parser.add_argument("--config", type=str, help="config path")
parser.add_argument("--ckpt", type=str, help="checkpoint path")
parser.add_argument("--patch_size", type=int, default=1024)
parser.add_argument("--stride", type=int, default=512)

def get_wsi_mask_path(wsi_path: str):
    """Get wsi-level for Mice glomeruli data
    Data structure:
    WSI data: test/wsi_id.tiff
    WSI mask: test_mask/wsi_id.tiff
    """
    mask_path = wsi_path.replace('/test/', '/test_mask/')
    if os.path.isfile(mask_path):
        return mask_path
    else:
        raise Exception(f'No ground-truth mask found for {wsi_path}!')
    
if __name__=="__main__":
    args = parser.parse_args()
    print(args)

    # define test_pipeline
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]

    # load model
    model = init_model(args.config, args.ckpt)
    # assign test_pipeline
    model.cfg.test_pipeline = test_pipeline
    print(model.cfg.model.backbone.type)

    mean_Dice = 0.0

    wsi_path = args.input
    wsi_name = Path(wsi_path).stem

    # Data is already in RGB
    wsi_data = tifffile.imread(wsi_path, key=0)
    H, W, _ = wsi_data.shape
    # load mask data
    mask_path = get_wsi_mask_path(wsi_path)
    mask_data = tifffile.imread(mask_path, key=0)

    assert (H, W) == mask_data.shape, f'WSI and GT mask not the same shape. {(H, W)} != {mask_data.shape}'

    ### Removing non-tissue areas
    wsi_data_resized = utils.rescale_wsi(wsi_data)
    wsi_filter_mask = utils.wsi_thresholding(wsi_data_resized)
    # resize threshold image back to original size
    wsi_filter_mask = cv2.resize(wsi_filter_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
    # mask out the tissue area
    wsi_data = cv2.bitwise_and(wsi_data, wsi_data, mask=wsi_filter_mask)
    
    # normalize mask data to [0, 1]
    mask_data = mask_data/mask_data.max()
    mask_data = mask_data.astype(np.uint8)

    # striding and predict
    wsi_shape = [2, H, W]

    x_slide = int((W - args.patch_size) / args.stride) + 1
    y_slide = int((H - args.patch_size) / args.stride) + 1

    # save predictions from all models
    pred_wsi_data = torch.full(wsi_shape, 0, dtype=torch.float)

    pbar = tqdm(range(x_slide*y_slide), leave=True)
    pbar.set_description(f'{wsi_name}')

    for xi in range(x_slide):
        for yi in range(y_slide):
            # update progress bar
            pbar.update(1)

            if xi == x_slide - 1:
                x_min = W - args.patch_size
            else:
                x_min = xi * args.stride

            if yi == y_slide - 1:
                y_min = H - args.patch_size
            else:
                y_min = yi * args.stride

            sub_wsi = wsi_data[y_min:y_min + args.patch_size, x_min:x_min + args.patch_size, :]
            # convert RGB to BGR
            sub_wsi = cv2.cvtColor(sub_wsi, cv2.COLOR_RGB2BGR)

            assert sub_wsi.shape == (1024, 1024, 3), f'Wrong shape {sub_wsi.shape}'
            # skip if image is non-tissue (i.e., all black)
            if len(np.unique(sub_wsi))==1:
                continue
            
            # predict
            pred_result = inference_model(model, sub_wsi)
            raw_logits = pred_result.seg_logits.data
            # softmax
            raw_logits = torch.softmax(raw_logits, dim=0)
            raw_logits = raw_logits.cpu()

            # store raw predictions
            pred_wsi_data[:, y_min:y_min + args.patch_size, x_min:x_min + args.patch_size] += raw_logits

    # normalize with softmax
    pred_wsi_data = torch.softmax(pred_wsi_data, dim=0)
    
    # get the predicted mask
    _, pred_seg = pred_wsi_data.max(axis=0, keepdims=True)
    pred_seg = pred_seg.cpu().numpy()[0]
    pred_seg = pred_seg.astype(np.uint8)

    # calculate DICE score
    dice_score = utils.calculate_dice(y_pred=pred_seg, y_gt=mask_data)
    print(f'{wsi_name} - Dice: {dice_score}')