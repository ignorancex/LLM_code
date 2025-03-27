import os
import sys
sys.path.append('./')
os.environ['TORCH_HOME'] = 'checkpoints'

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import joblib
import argparse

from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.metrics import adapted_rand_error
from scipy.spatial.distance import directed_hausdorff

import warnings
warnings.filterwarnings("ignore")

import torch
from utils.stats_utils import get_pq, get_multi_pq_info, get_multi_r2
from models.model_head_aug import HoVerNetHeadExt
from utils.visualize_gt import visualize
from utils.eval_utils import prepare_ground_truth, prepare_results, convert_pytorch_checkpoint

import numpy as np
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error
from scipy.spatial.distance import directed_hausdorff


def match_instances(gt_instance_map, pred_instance_map, threshold=0.3):

    matched_pairs = []
    unmatched_gt = []
    unmatched_pred = []

    gt_props = np.unique(gt_instance_map)
    pred_props = np.unique(pred_instance_map)

    for gt_id in gt_props:

        if gt_id == 0:
            continue
        gt_mask = gt_instance_map == gt_id

        best_match = None
        best_match_overlap = 0

        for pred_id in pred_props:
            if pred_id == 0:
                continue
            pred_mask = pred_instance_map == pred_id

            overlap = np.sum(np.logical_and(gt_mask, pred_mask)) / np.sum(np.logical_or(gt_mask, pred_mask))

            if overlap > threshold and overlap > best_match_overlap:
                best_match = pred_id
                best_match_overlap = overlap

        if best_match is not None:
            matched_pairs.append((gt_id, best_match))
        else:
            unmatched_gt.append(gt_id)

    for pred_id in pred_props:
        if pred_id not in [pair[1] for pair in matched_pairs]:
            unmatched_pred.append(pred_id)

    return matched_pairs, unmatched_gt, unmatched_pred

def calculate_hausdorff_distance(gt_instance_map, pred_instance_map, threshold=0.5):
    gt_boundaries = find_boundaries(gt_instance_map)
    pred_boundaries = find_boundaries(pred_instance_map)

    matched_pairs, unmatched_gt, unmatched_pred = match_instances(gt_instance_map, pred_instance_map, threshold)
    
    distances = []

    for gt_id, pred_id in matched_pairs:
        gt_mask = gt_boundaries * (gt_instance_map == gt_id)
        pred_mask = pred_boundaries * (pred_instance_map == pred_id)

        gt_points = np.argwhere(gt_mask)
        pred_points = np.argwhere(pred_mask)

        # 计算 Hausdorff 距离
        dist1 = directed_hausdorff(gt_points, pred_points)[0]
        dist2 = directed_hausdorff(pred_points, gt_points)[0]
        distance = max(dist1, dist2)

        distances.append(distance)

    distances = np.array(distances)

    gt_points = np.argwhere(gt_boundaries)
    pred_points = np.argwhere(pred_boundaries)

        # 计算 Hausdorff 距离
    dist1 = directed_hausdorff(gt_points, pred_points)[0]
    dist2 = directed_hausdorff(pred_points, gt_points)[0]

    if len(distances) > 0:
        return np.max(distances), max(dist1, dist2), unmatched_gt, unmatched_pred
    else:
        return np.nan, max(dist1, dist2), unmatched_gt, unmatched_pred
    # return max(dist1, dist2), None, None 

def eval_func(true_array, pred_array, true_csv, pred_csv, out_dir, epoch_idx, num_types=0):

    all_metrics = {}

    hausdorff_matched_list = []
    hausdorff_list = []

    pq_list = []
    mpq_info_list = []
    nr_patches = pred_array.shape[0]

    for patch_idx in tqdm(range(nr_patches)):
        # get a single patch
        pred = pred_array[patch_idx]
        true = true_array[patch_idx]

        # instance segmentation map
        pred_inst = pred[..., 0]
        true_inst = true[..., 0]

        # ===============================================================
        hausdorff_dis_matched, hausdorff_dis, _, _ = calculate_hausdorff_distance(pred_inst, true_inst)

        if not (np.isnan(hausdorff_dis_matched) or np.isinf(hausdorff_dis_matched)):
            hausdorff_matched_list.append(hausdorff_dis_matched)

        if not (np.isnan(hausdorff_dis) or np.isinf(hausdorff_dis)):
            hausdorff_list.append(hausdorff_dis)

        pq = get_pq(true_inst, pred_inst)
        pq = pq[0][2]
        pq_list.append(pq)

        # get the multiclass pq stats info from single image
        mpq_info_single = get_multi_pq_info(true, pred, nr_classes=num_types-1)
        mpq_info = []
        # aggregate the stat info per class
        for single_class_pq in mpq_info_single:
            tp = single_class_pq[0]
            fp = single_class_pq[1]
            fn = single_class_pq[2]
            sum_iou = single_class_pq[3]
            mpq_info.append([tp, fp, fn, sum_iou])
        mpq_info_list.append(mpq_info)

    pq_metrics = np.array(pq_list)
    pq_metrics_avg = np.mean(pq_metrics, axis=-1)  # average over all images

    hausdorff_metrics = np.array(hausdorff_list)
    hausdorff_metrics_avg = np.mean(hausdorff_metrics)

    hausdorff_matched_list = np.array(hausdorff_matched_list)
    hausdorff_matched_avg = np.mean(hausdorff_matched_list)
    mpq_info_metrics = np.array(mpq_info_list, dtype="float")
    # sum over all the images
    total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

    mpq_list = []
    # for each class, get the multiclass PQ
    for cat_idx in range(total_mpq_info_metrics.shape[0]):
        total_tp = total_mpq_info_metrics[cat_idx][0]
        total_fp = total_mpq_info_metrics[cat_idx][1]
        total_fn = total_mpq_info_metrics[cat_idx][2]
        total_sum_iou = total_mpq_info_metrics[cat_idx][3]

        # get the F1-score i.e DQ
        dq = total_tp / ((total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6)
                                    
        # get the SQ, when not paired, it has 0 IoU so does not impact
        sq = total_sum_iou / (total_tp + 1.0e-6)
        mpq_list.append(dq * sq)

    mpq_metrics = np.array(mpq_list)
    all_metrics['hausdorff_matched'] = [hausdorff_matched_avg]
    all_metrics['hausdorff'] = [hausdorff_metrics_avg]
    
    all_metrics["pq"] = [pq_metrics_avg]
    all_metrics["multi_pq+"] = [np.mean(mpq_metrics)]

    # first check to make sure ground truth and prediction is in csv format
    r2, r2_array = get_multi_r2(true_csv, pred_csv, return_array=True)
    all_metrics["multi_r2"] = [r2]

    if num_types == 7:
        nucleus_types = ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"]
    elif num_types == 6:
        nucleus_types = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    elif num_types == 5:
        nucleus_types = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]    


    for tp_idx in range(num_types-1):
        all_metrics[f"multi_pq_{nucleus_types[tp_idx]}"] = mpq_metrics[tp_idx]

    for tp_idx in range(num_types-1):
        all_metrics[f"multi_r2_{nucleus_types[tp_idx]}"] = r2_array[tp_idx]

    df = pd.DataFrame(all_metrics)
    os.makedirs(f"{out_dir}/results", exist_ok=True)
    df = df.to_csv(f"{out_dir}/results/{epoch_idx}.csv", index=False)


def eval_models(FOLD_IDX, imgs_load, labels, tp_num, exp_name, encoder_name, epoch_idx=30):
    valid_indices = range(len(imgs_load))

    checkpoint_path = f"checkpoints/{exp_name}/improved-net_{epoch_idx}.pt"
    segmentation_model = HoVerNetHeadExt(num_types=tp_num, encoder_name=encoder_name, pretrained_backbone=None)
    
    state_dict = torch.load(checkpoint_path)
    segmentation_model.load_state_dict(state_dict)
    segmentation_model = segmentation_model.to(0)
    segmentation_model.eval()
    print(f"============================{segmentation_model.training}=====================")

    np_results, hv_results, tp_results = [], [], []

    imgs_valid = imgs_load[valid_indices]
    for idx, img in tqdm(zip(valid_indices, imgs_valid), total=len(valid_indices)):
        img = img[None, :, :, :] / 255.
        img = torch.tensor(img)
        np_map, hv_map, tp_map = segmentation_model.infer_batch_inner_ensemble(segmentation_model, img, True)
        np_map = np_map[0]
        hv_map = hv_map[0]
        tp_map = tp_map[0]

        tp_map = np.argmax(tp_map, axis=-1)
        tp_map = np.array(tp_map, np.float32)
        tp_map = tp_map[:, :, None]

        np_results.append(np_map)
        hv_results.append(hv_map)
        tp_results.append(tp_map)

    labels_array_pred, nuclei_counts_df_pred, nuclei_counts_array_pred = prepare_results(np_results, hv_results, tp_results, segmentation_model)

    imgs_array_gt, labels_array_gt, nuclei_counts_df_gt, nuclei_counts_array_gt = prepare_ground_truth(imgs_load, labels, valid_indices)  

    # visualize(imgs_array_gt, labels_array_gt, labels_array_pred,f"visualize/overlay")
    eval_func(labels_array_gt, labels_array_pred, \
            nuclei_counts_df_gt, nuclei_counts_df_pred, f"wenhua_docker_test/{args.exp_name}_ensemble/{FOLD_IDX:02d}", epoch_idx, num_types=tp_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_idx', type=str, default='0')
    parser.add_argument('--exp_name', type=str, default='hover_paper_eight_axis_pannuke')
    parser.add_argument("--model", type=str, default="hovernet")
    parser.add_argument("--encoder_name", type=str, default="seresnext50_32x4d")
    args = parser.parse_args()

    tp_num = 7
    # hover_head_dropout_aug_glas
    if "monusac" in args.exp_name:
        img_path = "data_monusac/images_test.npy"
        ann_path = "data_monusac/labels_test.npy"
        tp_num = 5
    elif "pannuke" in args.exp_name:
        img_path = "data_pannuke/images_test.npy"
        ann_path = "data_pannuke/labels_test.npy"
        tp_num = 6

    labels = np.load(ann_path)
    imgs_load = np.load(img_path)

    for epoch_idx in range(31, 50):
        eval_models(0, imgs_load, labels, tp_num, args.exp_name, encoder_name=args.encoder_name, epoch_idx=epoch_idx)
