import os
import glob
from pathlib import Path
import cv2
import numpy as np

import torch
from mmseg.apis import init_model, inference_model

import utils
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="can be a single input or a directory of images from a WSI")
parser.add_argument("--config", type=str, help="config path")
parser.add_argument("--ckpt", type=str, help="checkpoint path")
parser.add_argument("--stitch", action="store_true", help="apply stitching strategy or not")
parser.add_argument("--img_size", type=int, help="2048 (KPIs) or 1024 (Mice glomeruli)")

def get_mask_path(img_path: str):
    """Get mask path for both KPIs (in .JPG) and Mice glomeruli (in .PNG) datasets
    """
    # in case of Mice glomeruli (orbit)
    mask_path = img_path.replace('/img/', '/mask/').replace('_img.jpg', '_mask.png')
    if os.path.isfile(mask_path):
        return mask_path
    else:
        # in case of KPIs
        mask_path = mask_path.replace('_mask.png', '_mask.jpg')
        if os.path.isfile(mask_path):
            return mask_path
        else:
            raise Exception(f'No mask found for {img_path}')


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

    # get image paths
    if os.path.isdir(args.input):
        # get WSI test data
        input_dir = Path(args.input)
        all_img_paths = glob.glob(str(Path(input_dir)/'**/*_img.*'), recursive=True)
    elif os.path.isfile(args.input):
        all_img_paths = [args.input]
    
    # check if stitching strategy can be performed
    is_stitching = args.stitch
    if is_stitching and len(all_img_paths) == 1:
        print(f'Found 1 input path, cannot perform stitching!')
        is_stitching = False

    print(f'Number of input: {len(all_img_paths)}')
    all_wsi_ids, all_coords = utils.get_wsi_data(all_img_paths, args.img_size)

    if len(set(all_wsi_ids)) > 1:
        print(f'Images in {args.input} are not from the same WSI, cannot perform stitching!')
        is_stitching = False

    mDice = 0.0
    # apply stitching strategy
    if is_stitching:
        print(f'Performing stitching strategy')
        all_coords = np.array(all_coords)

        # get the max of x and y [x_min, y_min, x_max, y_max]
        max_x = np.max(all_coords[:, 2])
        max_y = np.max(all_coords[:, 3])

        min_size = args.img_size
        if max_x < min_size:
            max_x = min_size
        if max_y < min_size:
            max_y = min_size

        # create a WSI binary predicted mask
        wsi_shape = [2, max_y, max_x]
        pred_wsi_data = torch.full(wsi_shape, 0, dtype=torch.float)

        pbar = tqdm(list(zip(all_img_paths, all_coords)), leave=True)
        for img_path, coord in pbar:
            img_data = cv2.imread(img_path, -1)
            x_min, y_min, x_max, y_max = coord

            # predict
            pred_res = inference_model(model, img_data)
            raw_logits = pred_res.seg_logits.data
            # softmax
            raw_logits = torch.softmax(raw_logits, dim=0)
            raw_logits = raw_logits.cpu()

            # store raw predictions
            pred_wsi_data[:, y_min:y_max, x_min:x_max] += raw_logits

        # get predicted mask from raw data
        pbar = tqdm(list(zip(all_img_paths, all_coords)), leave=True)
        print("Cropping back: ")
        for img_path, coord in pbar:
            # get mask data
            mask_path = get_mask_path(img_path)
            mask_data = cv2.imread(mask_path, -1)
            
            x_min, y_min, x_max, y_max = coord
            crop_pred_raw = pred_wsi_data[:, y_min:y_max, x_min:x_max]

            crop_pred_raw = torch.softmax(crop_pred_raw, dim=0)

            # get predicted mask
            pred_max_value, pred_seg = crop_pred_raw.max(axis=0, keepdims=True)

            pred_max_value = pred_max_value.cpu().numpy()[0]
            pred_seg = pred_seg.cpu().numpy()[0]

            # calculate DICE score
            dice_score = utils.calculate_dice(y_pred=pred_seg, y_gt=mask_data)
            mDice += dice_score

        print(f'Mean Dice: {mDice/len(all_img_paths)}')

    # segmentation per image patch
    else:
        print(f'Performing per image patch segmentation')
        for img_path in tqdm(all_img_paths):
            img_data = cv2.imread(img_path, -1)
            # get mask data
            mask_path = get_mask_path(img_path)
            mask_data = cv2.imread(mask_path, -1)
            
            # predict
            pred_res = inference_model(model, img_data)
            raw_logits = pred_res.seg_logits.data

            # get predicted mask
            _, pred_seg = raw_logits.max(axis=0, keepdims=True)
            pred_seg = pred_seg.cpu().numpy()[0]

            # calculate DICE score
            dice_score = utils.calculate_dice(y_pred=pred_seg, y_gt=mask_data)
            mDice += dice_score

        print(f'Mean Dice: {mDice/len(all_img_paths)}')