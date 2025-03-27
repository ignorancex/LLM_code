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

import warnings
warnings.filterwarnings("ignore")

import torch
from utils.stats_utils import get_pq, get_multi_pq_info, get_multi_r2
from models.model_head_aug import HoVerNetHeadExt
from utils.eval_utils import prepare_ground_truth, prepare_results, convert_pytorch_checkpoint
from torchmetrics.functional import dice    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_func(true_array, pred_array, true_csv, pred_csv, out_dir, epoch_idx, num_types=0):

    all_metrics = {}

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
    all_metrics["pq"] = [pq_metrics_avg]

    all_metrics["multi_pq+"] = [np.mean(mpq_metrics)]

    # first check to make sure ground truth and prediction is in csv format
    r2, r2_array = get_multi_r2(true_csv, pred_csv, return_array=True)
    all_metrics["multi_r2"] = [r2]

    cell_dice_list = []
    for i in range(1, 7):
        pred_array = torch.tensor(pred_array)
        true_array = torch.tensor(true_array)
        cell_dice = dice(preds=(pred_array[:, :, :, 1] == i), target=(true_array[:, :, :, 1] == i), ignore_index=0).cpu()
        cell_dice_list.append(cell_dice.numpy())
    
    all_metrics['dice'] = np.mean(np.array(cell_dice_list))
    

    if num_types == 7:
        all_metrics["multi_pq_neutrophil"] = mpq_metrics[0]
        all_metrics["multi_pq_epithelial"] = mpq_metrics[1]
        all_metrics["multi_pq_lymphocyte"] = mpq_metrics[2]
        all_metrics["multi_pq_plasma"] = mpq_metrics[3]
        all_metrics["multi_pq_eosinophil"] = mpq_metrics[4]
        all_metrics["multi_pq_connective"] = mpq_metrics[5]

        all_metrics["multi_r2_neutrophil"] = r2_array[0]
        all_metrics["multi_r2_epithelial"] = r2_array[1]
        all_metrics["multi_r2_lymphocyte"] = r2_array[2]
        all_metrics["multi_r2_plasma"] = r2_array[3]
        all_metrics["multi_r2_eosinophil"] = r2_array[4]
        all_metrics["multi_r2_connective"] = r2_array[5]

    df = pd.DataFrame(all_metrics)
    os.makedirs(f"{out_dir}/results", exist_ok=True)
    df = df.to_csv(f"{out_dir}/results/{epoch_idx}.csv", index=False)


def eval_models(log_name, FOLD_IDX, imgs_load, labels, tp_num, exp_name0, encoder_name0, exp_name1, encoder_name1, epoch_idx=30):
    splits = joblib.load("splits_10_per_dataset.dat")
    valid_indices = splits[FOLD_IDX]['valid']

    checkpoint_path0 = f"checkpoints/{exp_name0}/improved-net_{epoch_idx}.pt"
    segmentation_model0 = HoVerNetHeadExt(num_types=7, encoder_name=encoder_name0, pretrained_backbone=None)

    checkpoint_path1 = f"checkpoints/{exp_name1}/improved-net_{epoch_idx}.pt"
    segmentation_model1 = HoVerNetHeadExt(num_types=7, encoder_name=encoder_name1, pretrained_backbone=None)
    
    print(f"===================parameter counts: {count_parameters(segmentation_model0) + count_parameters(segmentation_model1)}=====")
    print(f"===================parameter counts: {count_parameters(segmentation_model0)} ")
    print(f"===================parameter counts: {count_parameters(segmentation_model1)} ")
    state_dict = torch.load(checkpoint_path0)
    segmentation_model0.load_state_dict(state_dict)
    segmentation_model0 = segmentation_model0.to(0)
    segmentation_model0.eval()
    
    state_dict = torch.load(checkpoint_path1)
    segmentation_model1.load_state_dict(state_dict)
    segmentation_model1 = segmentation_model1.to(0)
    segmentation_model1.eval()

    np_results, hv_results, tp_results = [], [], []

    imgs_valid = imgs_load[valid_indices]

    for idx, img in tqdm(zip(valid_indices, imgs_valid), total=len(valid_indices)):
        img = img[None, :, :, :] / 255.
        img = torch.tensor(img)
        np_map0, hv_map0, tp_map0 = segmentation_model0.infer_batch_inner_ensemble(segmentation_model0, img, True, idx=idx, encoder_name="seresnext50")
        np_map1, hv_map1, tp_map1 = segmentation_model1.infer_batch_inner_ensemble(segmentation_model1, img, True, idx=idx, encoder_name="seresnext101")
        
        np_map = (np_map0[0] + np_map1[0]) / 2
        hv_map = (hv_map0[0] + hv_map1[0]) / 2
        tp_map = (tp_map0[0] + tp_map1[0]) / 2

        tp_map = np.argmax(tp_map, axis=-1)
        tp_map = np.array(tp_map, np.float32)
        tp_map = tp_map[:, :, None]

        np_results.append(np_map)
        hv_results.append(hv_map)
        tp_results.append(tp_map)

    labels_array_pred, nuclei_counts_df_pred, nuclei_counts_array_pred = prepare_results(np_results, hv_results, tp_results, segmentation_model0)

    imgs_array_gt, labels_array_gt, nuclei_counts_df_gt, nuclei_counts_array_gt = prepare_ground_truth(imgs_load, labels, valid_indices)  

    eval_func(labels_array_gt, labels_array_pred, \
            nuclei_counts_df_gt, nuclei_counts_df_pred, f"wenhua_docker_test/{log_name}/{exp_name0}/{FOLD_IDX:02d}", epoch_idx, num_types=tp_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_idx', type=str, default='0')
    parser.add_argument("--model", type=str, default="hovernet")
    parser.add_argument("--log_name", type=str, default="ensemble_all_fold_0_conic")
    
    parser.add_argument('--exp_name0', type=str, default='hover_paper_conic_seresnext50_00')
    parser.add_argument("--encoder_name0", type=str, default="seresnext50")

    parser.add_argument('--exp_name1', type=str, default='hover_paper_conic_seresnext101_00')
    parser.add_argument("--encoder_name1", type=str, default="seresnext101")

    args = parser.parse_args()

    num_types = 7
    # hover_head_dropout_aug_glas
    img_path = "data/images.npy"
    ann_path = "data/labels.npy"
    labels = np.load(ann_path)
    imgs_load = np.load(img_path)

    folder_idx = int(args.folder_idx)
    
    epoch_idx = 49
    eval_models(args.log_name, folder_idx, imgs_load, labels, 7, \
                args.exp_name0, args.encoder_name0, \
                args.exp_name1, args.encoder_name1, epoch_idx=epoch_idx)
