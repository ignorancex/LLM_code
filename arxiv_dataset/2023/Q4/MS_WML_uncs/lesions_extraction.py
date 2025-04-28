from scipy import ndimage
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from .lesion_uncertainty_measures import intersection_over_union


def get_tp_fp_fn_lesions(y_pred: np.ndarray, y: np.ndarray, IoU_threshold: float, n_jobs: int = None) -> tuple:
    """
    Get binary lesion masks with TP, FP and FN lesions separately.
    :param y_pred: predicted binary lesion mask, shape [H, W, D]
    :param y: gt binary lesion mask, shape [H, W, D]
    :param IoU_threshold: if lesions have IoU > threshold, then they are TP, otherwise FP
    :param n_jobs: number of parallel processes
    :return: binary lesion masks with TP, FP and FN lesions separately
    :rtype: tuple
    """
    def get_tp_fp(label_pred, gt_multi, pred_multi):
        lesion_pred = (pred_multi == label_pred).astype(float)
        max_iou = 0.0
        for label_gt in np.unique(gt_multi * lesion_pred):  # iterate only intersections
            if label_gt != 0.0:
                mask_label_gt = (gt_multi == label_gt).astype(int)
                iou = intersection_over_union(lesion_pred, mask_label_gt)
                if iou > max_iou:
                    max_iou = iou
        return "tp" if max_iou >= IoU_threshold else "fp", label_pred

    def get_fn(label_gt, pred_bin, gt_multi):
        lesion_gt = (gt_multi == label_gt).astype(float)
        iou = intersection_over_union(lesion_gt, pred_bin)
        if iou == 0.0:
            return label_gt

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    gt_multi_mask, n_gt_labels = ndimage.label(y, structure=struct_el)
    pred_multi_mask, n_pred_labels = ndimage.label(y_pred, structure=struct_el)

    process_tp_fn = partial(get_tp_fp, gt_multi=gt_multi_mask, pred_multi=pred_multi_mask)
    process_fn = partial(get_fn, pred_bin=y_pred, gt_multi=gt_multi_mask)

    tp_lesions, fp_lesions = [], []

    with Parallel(n_jobs=n_jobs) as parallel:
        fn_lesions = parallel(delayed(process_fn)(label) for label in range(1, n_gt_labels + 1))
        fn_lesions = [_ for _ in fn_lesions if _ is not None]

        tps_fps = parallel(delayed(process_tp_fn)(label) for label in range(1, n_pred_labels + 1))
        for lesion_type, lesion in tps_fps:
            if lesion_type == 'fp':
                fp_lesions.append(lesion)
            else:
                tp_lesions.append(lesion)

    return np.isin(pred_multi_mask, tp_lesions), \
           np.isin(pred_multi_mask, fp_lesions), \
           np.isin(gt_multi_mask, fn_lesions)


def gt_fn_count(y_pred: np.ndarray, y: np.ndarray, n_jobs: int = None) -> float:
    """
    Get number of FN lesions (no intersection with predictions) on the ground truth binary mask.
    :param y_pred: predicted binary lesion mask, shape [H, W, D]
    :param y: gt binary lesion mask, shape [H, W, D]
    :param n_jobs: number of parallel processes
    :return: number of false negative lesions in a scan
    """
    def get_fn(label_gt, pred_bin, gt_multi):
        lesion_gt = (gt_multi == label_gt).astype(float)
        iou = intersection_over_union(lesion_gt, pred_bin)
        if iou == 0.0:
            return label_gt

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    gt_multi_mask, n_gt_labels = ndimage.label(y, structure=struct_el)

    process = partial(get_fn, pred_bin=y_pred, gt_multi=gt_multi_mask)
    fn_lesions = Parallel(n_jobs=n_jobs)(delayed(process)(label_gt) for label_gt in range(1, n_gt_labels + 1))
    return np.sum([les for les in fn_lesions if les is not None])
