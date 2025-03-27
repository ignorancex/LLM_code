import os
import sys
sys.path.append("./")
import cv2

import numpy as np
from utils.visualize_gt import draw_dilation_seg

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def pannuke(split=0, test=False):
    print("========================================")
    
    if test:
        split = (split + 2) % 3

    images_train = []
    labels_train = []

    images_this = np.load(f"/data/teacher/workspace/csbj/HoVerNet-JBHI/data_pannuke/panNuke/Fold_{split+1}/images/images.npy")
    masks_this = np.load(f"/data/teacher/workspace/csbj/HoVerNet-JBHI/data_pannuke/panNuke/Fold_{split+1}/masks/masks.npy")

    for img_single, mask_single in zip(images_this, masks_this):
        images_train.append(img_single)

        inst_map = np.zeros((256, 256))
        num_nuc = 0
        for j in range(5):
            # copy value from new array if value is not equal 0
            layer_res = remap_label(mask_single[:, :, j])
            # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            num_nuc = num_nuc + np.max(layer_res)
        inst_map = remap_label(inst_map)

        type_map = np.zeros((256, 256)).astype(np.int32)
        for j in range(5):
            layer_res = ((j + 1) * np.clip(mask_single[:, :, j], 0, 1)).astype(np.int32)
            type_map = np.where(layer_res != 0, layer_res, type_map)
                
        inst_type = np.stack((inst_map, type_map), axis=-1)
        labels_train.append(inst_type)
    
    images_train = np.array(images_train)
    print("images_train shape: ", images_train.shape)

    labels_train = np.array(labels_train)
    print("labels_train shape: ", labels_train.shape)

    os.makedirs(f"/data/teacher/workspace/csbj/HoVerNet-JBHI/data_pannuke/split_{split}/", exist_ok=True)

    if test:
        train_test = "test"
    else:
        train_test = "train"

    # images_to_lean = np.load(f"/data/teacher/workspace/csbj/HoVerNet-JBHI/data_pannuke/images_{train_test}.npy")
    # print("images_to_learn shape: ", images_to_lean.shape)

    np.save(f"/data/teacher/workspace/csbj/HoVerNet-JBHI/data_pannuke/split_{split}/images_{train_test}.npy", images_train)
    np.save(f"/data/teacher/workspace/csbj/HoVerNet-JBHI/data_pannuke/split_{split}/labels_{train_test}.npy", labels_train)
    
                
                

if __name__ == "__main__":
    pannuke(split=0, test=False)
    pannuke(split=1, test=False)
    pannuke(split=2, test=False)

    pannuke(split=0, test=True)
    pannuke(split=1, test=True)
    pannuke(split=2, test=True)
    