import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from typing import Tuple, Union
from termcolor import colored
import shutil

import warnings
warnings.filterwarnings("ignore")

from utils.stats_utils import get_pq, get_multi_pq_info, get_multi_r2, remap_label


def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array

    Returns:
        x: cropped array

    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    """Remove and then make a new directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

####
def overlay_prediction_contours(
    canvas: np.ndarray,
    inst_dict: dict,
    draw_dot: bool = False,
    type_colours: dict = None,
    inst_colours: Union[np.ndarray, Tuple[int]] = (255, 255, 0),
    line_thickness: int = 2,
):
    """Overlaying instance contours on image.

    Internally, colours from `type_colours` are prioritized over
    `inst_colours`. However, if `inst_colours` is `None` and `type_colours`
    is not provided, random colour is generated for each instance.

    Args:
        canvas (ndarray): Image to draw predictions on.
        inst_dict (dict): Dictionary of instances. It is expected to be
            in the following format:
            {instance_id: {type: int, contour: List[List[int]], centroid:List[float]}.
        draw_dot (bool): To draw a dot for each centroid or not.
        type_colours (dict): A dict of {type_id : (type_name, colour)},
            `type_id` is from 0-N and `colour` is a tuple of (R, G, B).
        inst_colours (tuple, np.ndarray): A colour to assign for all instances,
            or a list of colours to assigned for each instance in `inst_dict`. By
            default, all instances will have RGB colour `(255, 255, 0).
        line_thickness: line thickness of contours.

    Returns:
        (np.ndarray) The overlaid image.

    """
    overlay = np.copy((canvas))

    if isinstance(inst_colours, tuple):
        inst_colours = np.array([inst_colours] * len(inst_dict))
    elif not isinstance(inst_colours, np.ndarray):
        raise ValueError(
            f"`inst_colours` must be np.ndarray or tuple: {type(inst_colours)}"
        )
    inst_colours = inst_colours.astype(np.uint8)

    for idx, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colours is not None:
            inst_colour = type_colours[inst_info["type"]][1]
        else:
            inst_colour = (inst_colours[idx]).tolist()
        cv2.drawContours(
            overlay, [np.array(inst_contour)], -1, inst_colour, line_thickness
        )

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def process_segmentation(np_map, hv_map, tp_map, model):
    # HoVerNet post-proc is coded at 0.25mpp so we resize
    np_map = cv2.resize(np_map, (0, 0), fx=2.0, fy=2.0)
    hv_map = cv2.resize(hv_map, (0, 0), fx=2.0, fy=2.0)
    tp_map = cv2.resize(tp_map, (0, 0), fx=2.0, fy=2.0,
                        interpolation=cv2.INTER_NEAREST)

    inst_map = model._proc_np_hv(np_map[..., None], hv_map)
    inst_dict = model._get_instance_info(inst_map, tp_map)

    # Generating results match with the evaluation protocol
    type_map = np.zeros_like(inst_map)
    inst_type_colours = np.array([
            [v['type']] * 3 for v in inst_dict.values()
        ])

    type_map = overlay_prediction_contours(
            type_map, inst_dict,
            line_thickness=-1,
            inst_colours=inst_type_colours)
            
    pred_map = np.dstack([inst_map, type_map])
    # The result for evaluation is at 0.5mpp so we scale back
    pred_map = cv2.resize(pred_map, (0, 0), fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_NEAREST)
    return pred_map


def process_composition(pred_map, NUM_TYPES=7):
    # Only consider the central 224x224 region,
    # as noted in the challenge description paper
    pred_map = cropping_center(pred_map, [224, 224])
    inst_map = pred_map[..., 0]
    type_map = pred_map[..., 1]
    # ignore 0-th index as it is 0 i.e background
    uid_list = np.unique(inst_map)[1:]

    if len(uid_list) < 1:
        type_freqs = np.zeros(NUM_TYPES)
        return type_freqs
    uid_types = [
        np.unique(type_map[inst_map == uid])
        for uid in uid_list
        ]
    type_freqs_ = np.unique(uid_types, return_counts=True)
    # ! not all types exist within the same spatial location
    # ! so we have to create a placeholder and put them there
    type_freqs = np.zeros(NUM_TYPES)
    type_freqs[type_freqs_[0]] = type_freqs_[1]
    return type_freqs


def get_npy_csv(masks, patch_shape=[256, 256]):
    inst_map_list = []
    class_map_list = []
    nuclei_counts_list = []
    for ann in masks:
        patch_inst = ann[..., 0]  # instance map
        patch_class = ann[..., 1]  # class map

        # ensure nuclei range from 0 to N (N is the number of nuclei in the patch)
        patch_inst = remap_label(patch_inst)

        # only consider nuclei for counting if it exists within the central 224x224 region
        if patch_inst.shape[0] == 256:
            patch_inst_crop = cropping_center(patch_inst, [224, 224])
            patch_class_crop = cropping_center(patch_class, [224, 224])
        elif patch_inst.shape[0] == 224:
            tgt_crop_size = int( 224 * (224 / 256))
            patch_inst_crop = cropping_center(patch_inst, [tgt_crop_size, tgt_crop_size])
            patch_class_crop = cropping_center(patch_class, [tgt_crop_size, tgt_crop_size])
        
        nuclei_counts_perclass = []
        # get the counts per class
        for nuc_val in range(1, 7):
            patch_class_crop_tmp = patch_class_crop == nuc_val
            patch_inst_crop_tmp = patch_inst_crop * patch_class_crop_tmp
            nr_nuclei = len(np.unique(patch_inst_crop_tmp).tolist()[1:])
            nuclei_counts_perclass.append(nr_nuclei)

        if patch_inst.shape[0] != patch_shape[0]:
            patch_inst = cropping_center(patch_inst, patch_shape)
            patch_class = cropping_center(patch_class, patch_shape)

        inst_map_list.append(patch_inst)
        class_map_list.append(patch_class)
        nuclei_counts_list.append(nuclei_counts_perclass)

    # convert to numpy array
    inst_map_array = np.array(inst_map_list).astype("uint16")
    class_map_array = np.array(class_map_list).astype("uint16")
    nuclei_counts_array = np.array(nuclei_counts_list).astype("uint16")

    # combine instance map and classification map to form single array
    inst_map_array = np.expand_dims(inst_map_array, -1)
    class_map_array = np.expand_dims(class_map_array, -1)
    labels_array = np.concatenate((inst_map_array, class_map_array), axis=-1)

    # convert to pandas dataframe
    nuclei_counts_df = pd.DataFrame(
        data={
            "neutrophil": nuclei_counts_array[:, 0],
            "epithelial": nuclei_counts_array[:, 1],
            "lymphocyte": nuclei_counts_array[:, 2],
            "plasma": nuclei_counts_array[:, 3],
            "eosinophil": nuclei_counts_array[:, 4],
            "connective": nuclei_counts_array[:, 5],
        }
    )

    return labels_array, nuclei_counts_df, nuclei_counts_array


def prepare_ground_truth(imgs, masks, valid_indexes):
    masks_valid = masks[valid_indexes]
    imgs_valid = imgs[valid_indexes]
    labels_array_gt, nuclei_counts_df_gt, nuclei_counts_array_gt = get_npy_csv(masks_valid, patch_shape=[224, 224])

    return imgs_valid, labels_array_gt, nuclei_counts_df_gt, nuclei_counts_array_gt
    

def prepare_results(np_results, hv_results, tp_results, model):
    semantic_predictions = []
    for (np_map, hv_map, tp_map) in zip(np_results, hv_results, tp_results):
        np_map = np.array(np_map)
        hv_map = np.array(hv_map)
        tp_map = np.array(tp_map)

        pred_map = process_segmentation(np_map, hv_map, tp_map, model)

        # input("checkpoint")
        semantic_predictions.append(pred_map)
    
    semantic_predictions = np.array(semantic_predictions)

    labels_array_pred, nuclei_counts_df_pred, nuclei_counts_array_pred = get_npy_csv(masks=semantic_predictions,
                                                                                     patch_shape=[224, 224])
    return labels_array_pred, nuclei_counts_df_pred, nuclei_counts_array_pred


def eval(imgs, true_array, pred_array, nuclei_counts_gt, nuclei_counts_pred, true_csv, pred_csv, out_dir, epoch_idx, num_types=0):

    all_metrics = {}

    pq_list = []
    mpq_info_list = []
    nr_patches = pred_array.shape[0]

    # visualize(imgs, true_array, pred_array,f"{out_dir}/overlay")

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
        mpq_info_single = get_multi_pq_info(true, pred)
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