import os
import glob
from pathlib import Path

import cv2
import tifffile
import numpy as np

import json

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, help="path to mice glomerui dir")
parser.add_argument("--ann_root", type=str, help="path to annotation mask JSON dir")
parser.add_argument("--crop_size", type=int, default=1024)

def find_data_group(file_path: str, data_group: dict):
    is_test = False
    if '/test/' in file_path:
        is_test = True

    # convert file ID in data group to string
    new_data_group = data_group.copy()
    for key in new_data_group:
        new_data_group[key] = [str(x) for x in new_data_group[key]]
        
    file_id = Path(file_path).stem
    
    for key in new_data_group:
        if str(file_id) in new_data_group[key]:
            return is_test, key
    
    return None

def process_points(xpoints: list, ypoints: list):
    assert len(xpoints) == len(ypoints), f'xpoints and ypoints must have the same length!'
    new_xpoints = []
    new_ypoints = []

    for idx in range(len(xpoints)):
        # ignore point (0, 0)
        if xpoints[idx] != 0 and ypoints[idx] != 0:
            new_xpoints.append(xpoints[idx])
            new_ypoints.append(ypoints[idx])

    # close the polygon
    new_xpoints.append(new_xpoints[0])
    new_ypoints.append(new_ypoints[0])
    
    return np.array(new_xpoints), np.array(new_ypoints)

def get_anno_coordinate(anno: dict, wsi_shape: tuple, crop_size=1024):
    """Given an annotation data, get the coordinate for cropping
    Args:
        anno: an annotation dictionary
        wsi_shape: (H, W) of a WSI
        crop_size: center crop the annotation mask with this size
    Returns:
        coordinate (np.ndarray): [x_min, y_min, x_max, y_max]
    """
    H, W = wsi_shape
    xpoints = anno['xpoints']
    ypoints = anno['ypoints']

    new_xpoints, new_ypoints = process_points(xpoints, ypoints)

    x_c = (new_xpoints.min() + new_xpoints.max())//2
    y_c = (new_ypoints.min() + new_ypoints.max())//2
    
    # get cropped coordinates
    x_min = x_c - crop_size//2
    x_max = x_c + crop_size//2
    y_min = y_c - crop_size//2
    y_max = y_c + crop_size//2
    
    # process the coordinates
    if x_min < 0:
        x_min = 0
        x_max = x_min + crop_size
    if y_min < 0:
        y_min = 0
        y_max = y_min + crop_size
    if x_max > W:
        x_max = W
        x_min = y_max - crop_size
    if y_max > H:
        y_max = H
        y_min = y_max - crop_size

    return np.array([x_min, y_min, x_max, y_max])

if __name__=="__main__":
    # data group for data split
    data_group = {
        'FastRed_Mouse': ['6533664', '6533666', '6533668', '6533679', '6533672', '6533678', '6533683', '6533680', '6533682', '6533691', '6533687', '6533688'],
        'H_E_Mouse_G1': ['6654562', '6654559', '6654566', '6654568', '6654582', '6654586', '6654587', '6654588', '6654517'],
        'H_E_Mouse_G2': ['6479221', '6479222', '6479223', '6479224', '6479183', '6479186', '6479191', '6479195', '6479196', '6479203'],
        'H_E_Mouse_G3': ['6666707', '6666708', '6666709', '6666710', '6666711', '6666712', '6666713', '6666714', '6666717', '6666718'],

        'PAS_Rat': ['6609616', '6609615', '6609629', '6609617', '6609626', '6609628', '6625497', '6625501', '6625506', '6625505', '6609613', '6609634', '6609605'],
        'H_DAB_Rat_G1': ['5483162', '5482449', '5483170', '5482455', '5483190', '5482452', '5482458', '5483139', '5483117', '5482411', '5483132'],
        'H_DAB_Rat_G2': ['4737452', '4737489', '4737509', '4737522', '4730025', '4730043', '4730080', '4758050', '4758056', '4758065', '4758073'],
        'H_DAB_Rat_G3': ['6139966', '6139967', '6139977', '6139983', '6140227', '6140234', '6140254', '6140232', '6140251', '6140179', '6140168', '6140176'],
    }

    # not saving these WSI-level masks due to possibly incomplete in labeling
    exclusive_wsi_masks = [
        '6666717', '6533672', '6666713', '6533682', '6666711',
    ]

    args = parser.parse_args()
    print(args)

    seg_ann_root = Path(args.ann_root)
    seg_ann_paths = glob.glob(str(seg_ann_root/'*.json'))
    seg_ann_paths = sorted(seg_ann_paths)

    print(f'Found {len(seg_ann_paths)} labels')

    # where to save extracted patches
    extracted_patch_dir = Path(args.data_root)/'extracted_data'

    for ann_path in seg_ann_paths:
        # Open and read the JSON file
        with open(ann_path, 'r') as file:
            data = json.load(file)
        
        seg_data = data['seg_data']
        wsi_path = str(args.data_root) + data['data_path']
        print(wsi_path)

        # load WSI data
        wsi_data = tifffile.imread(wsi_path, key=0)
        H, W, _ = wsi_data.shape

        print(f'{wsi_data.shape=}')

        # exclude ROI and exclusion segmentation type
        new_seg_data = []
        for anno in seg_data:
            try:
                # only get annotation with normal type
                if anno['sub_type'] == 'Normal':
                    new_seg_data.append(anno)
            except:
                continue

        mask_data = np.zeros((H, W))

        # fill segmentation mask
        all_areas = []
        for anno in new_seg_data:
            xpoints = anno['xpoints']
            ypoints = anno['ypoints']

            new_xpoints, new_ypoints = process_points(xpoints, ypoints)

            ann_points = [p for p in zip(new_xpoints, new_ypoints)]
            all_areas.append(np.array(ann_points))

        # fill all annotation masks with value 1
        mask_data = cv2.fillPoly(mask_data, all_areas, 1)
        mask_data = mask_data.astype(np.uint8)

        # prepare to extract data
        is_test, wsi_group = find_data_group(file_path=wsi_path, data_group=data_group)
        
        if is_test:
            save_wsi_mask_dir = Path(args.data_root)/'test_mask'
            save_dir = extracted_patch_dir/'Testing_data_patch'
        else:
            save_wsi_mask_dir = Path(args.data_root)/'train_mask'
            save_dir = extracted_patch_dir/'Training_data_patch'

        # create WSI_group dir
        wsi_id = Path(wsi_path).stem
        save_mask_dir = save_dir/wsi_group/wsi_id/'mask'
        save_img_dir = save_dir/wsi_group/wsi_id/'img'
        
        os.makedirs(save_mask_dir, exist_ok=True)
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_wsi_mask_dir, exist_ok=True)

        count = 1
        pbar = tqdm(new_seg_data, leave=True)
        pbar.set_description(f"{wsi_group}/{wsi_id}")
        for anno in pbar:
            x_min, y_min, x_max, y_max = get_anno_coordinate(anno, wsi_shape=(H, W), crop_size=args.crop_size)

            # crop image
            sub_img = wsi_data[y_min:y_max, x_min:x_max, :]
            # convert to BGR for saving
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)
            # crop mask
            sub_mask = mask_data[y_min:y_max, x_min:x_max]

            filename = f'{wsi_group}_{wsi_id}_{count}_{x_min}_{y_min}'
            # save mask data
            cv2.imwrite(str(save_mask_dir/f'{filename}_mask.png'), sub_mask*255.)
            # save img data
            cv2.imwrite(str(save_img_dir/f'{filename}_img.jpg'), sub_img)
            
            count += 1

        # saving WSI mask data
        if wsi_id not in exclusive_wsi_masks:
            tifffile.imwrite(str(save_wsi_mask_dir/f'{Path(wsi_path).name}'), mask_data)