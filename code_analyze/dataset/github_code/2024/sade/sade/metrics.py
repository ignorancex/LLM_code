"""Metrics for evaluating OOD detection and segmentation performance"""

import logging
import math

import ants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.filters as skf
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
from scipy.stats import percentileofscore
from skimage.morphology import ball
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm


def ood_metrics(
    inlier_score, outlier_score, plot=False, verbose=False, names=["Inlier", "Outlier"]
):
    import numpy as np
    import seaborn as sns

    y_true = np.concatenate((np.zeros(len(inlier_score)), np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))

    prec_in, rec_in, _ = precision_recall_curve(y_true, y_scores)

    # Outliers are treated as "positive" class
    # i.e label 1 is now label 0
    prec_out, rec_out, _ = precision_recall_curve((y_true == 0), -y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)

    # rtol=1e-3 implies range of [0.949, 0.951]
    find_fpr = np.isclose(tpr, 0.95, rtol=1e-3, atol=1e-4).any()

    if find_fpr:
        tpr99_idx = np.where(np.isclose(tpr, 0.99, rtol=-1e-3, atol=1e-4))[0][0]
        tpr95_idx = np.where(np.isclose(tpr, 0.95, rtol=1e-3, atol=1e-4))[0][0]
    else:
        # This is becasuse numpy bugs out when the scores are fully separable
        # OR completely unseparable :D
        tpr99_idx = np.where(np.isclose(tpr, 0.99, rtol=1e-2, atol=1e-2))[0][0]
        # print("Clipping 99 TPR to:", tpr[tpr99_idx])
        if np.isclose(tpr, 0.95, rtol=-1e-2, atol=3e-2).any():
            tpr95_idx = np.where(np.isclose(tpr, 0.95, rtol=-1e-2, atol=3e-2))[0][0]
        else:
            tpr95_idx = np.where(np.isclose(tpr, 0.95, rtol=2e-2, atol=3e-2))[0][0]

        logging.info(f"Clipping 95 TPR to: {tpr[tpr95_idx]}")
        logging.info(f"Clipping 99 TPR to: {tpr[tpr99_idx]}")

    # Detection Error
    de = np.min(0.5 - tpr / 2 + fpr / 2)

    metrics = dict(
        true_tpr95=tpr[tpr95_idx],
        fpr_tpr99=fpr[tpr99_idx],
        fpr_tpr95=fpr[tpr95_idx],
        de=de,
        roc_auc=roc_auc_score(y_true, y_scores),
        pr_auc_in=auc(rec_in, prec_in),
        pr_auc_out=auc(rec_out, prec_out),
        #         fpr_tpr80=fpr[tpr80_idx],
        ap=average_precision_score(y_true, y_scores),
    )

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=True)
        ticks = np.arange(0.0, 1.1, step=0.1)

        axs[0].plot(fpr, tpr)
        axs[0].set(
            xlabel="FPR",
            ylabel="TPR",
            title="ROC",
            ylim=(-0.05, 1.05),
            xticks=ticks,
            yticks=ticks,
        )

        axs[1].plot(rec_in, prec_in, label="PR-In")
        axs[1].plot(rec_out, prec_out, label="PR-Out")
        axs[1].set(
            xlabel="Recall",
            ylabel="Precision",
            title="Precision-Recall",
            ylim=(-0.05, 1.05),
            xticks=ticks,
            yticks=ticks,
        )
        axs[1].legend()
        fig.suptitle("{} vs {}".format(*names), fontsize=20)
    #         plt.show()
    #         plt.close()

    if verbose:
        print("{} vs {}".format(*names))
        print("----------------")
        print("ROC-AUC: {:.4f}".format(metrics["roc_auc"] * 100))
        print(
            "PR-AUC (In/Out): {:.4f} / {:.4f}".format(
                metrics["pr_auc_in"] * 100, metrics["pr_auc_out"] * 100
            )
        )
        print("FPR (95% TPR): {:.2f}%".format(metrics["fpr_tpr95"] * 100))
        print("Detection Error: {:.2f}%".format(de * 100))
        print("FPR (99% TPR): {:.2f}%".format(metrics["fpr_tpr99"] * 100))

    return metrics


def plot_curves(inlier_score, outlier_score, label, axs=()):
    if len(axs) == 0:
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    y_true = np.concatenate((np.zeros(len(inlier_score)), np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=True)
    roc_auc = roc_auc = roc_auc_score(y_true, y_scores)

    prec_in, rec_in, _ = precision_recall_curve(y_true, y_scores)
    prec_out, rec_out, _ = precision_recall_curve((y_true == 0), -y_scores)
    pr_auc = auc(rec_in, prec_in)

    ticks = np.arange(0.0, 1.1, step=0.1)
    axs[0].plot(fpr, tpr, label="{}: {:.3f}".format(label, roc_auc))
    axs[0].set(
        xlabel="FPR",
        ylabel="TPR",
        title="ROC",
        ylim=(-0.05, 1.05),
        xticks=ticks,
        yticks=ticks,
    )

    axs[1].plot(rec_in, prec_in, label="{}: {:.3f}".format(label, pr_auc))
    # axs[1].plot(rec_out, prec_out, label="PR-Out")
    axs[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall",
        ylim=(-0.05, 1.05),
        xticks=ticks,
        yticks=ticks,
    )

    axs[0].legend()
    axs[1].legend()

    return axs


###### Segmentation metrics and helper functions ########


def segmentation_metrics(reference_ccp, segmentation_ccp):
    """Compute component-wise segmentation metrics for a single sample"""
    FP = 0
    TP = 0
    FN = 0

    GT = len(np.unique(reference_ccp.ravel())) - 1
    S = len(np.unique(segmentation_ccp.ravel())) - 1

    # Using reference components
    for i in range(1, GT + 1):
        lesion_comp_mask = reference_ccp == i
        seg_overlap = segmentation_ccp[lesion_comp_mask]
        if np.unique(seg_overlap).sum() > 0:
            TP += 1
        else:
            # No matching segementation component
            FN += 1

    # Iterating over segemntation components
    for i in range(1, S + 1):
        lesion_comp_mask = segmentation_ccp == i
        seg_overlap = reference_ccp[lesion_comp_mask]

        # If no matching reference component
        if np.unique(seg_overlap).sum() == 0:
            FP += 1

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "ground_truth_components": GT,
        "segmentation_components": S,
    }

    return metrics


def erode_brain_masks(masks, radius=2):
    if isinstance(masks, torch.Tensor):
        masks = masks.float().numpy()

    eroded_masks = np.zeros_like(masks)
    for i, m in enumerate(masks):
        m = ants.from_numpy(m)
        m = ants.morphology(
            m,
            operation="erode",
            radius=radius,
            mtype="binary",
            shape="ball",
            radius_is_parametric=True,
        )
        eroded_masks[i] = m.numpy()
    return eroded_masks


def remove_small_components(ccp, size=1, verbose=True):
    filtered_ccp = ccp.copy()
    n_components = len(np.unique(ccp)) - 1
    new_components = n_components
    for i in range(1, n_components + 1):
        lesion_comp_mask = ccp == i
        if lesion_comp_mask.sum() < size:
            filtered_ccp[lesion_comp_mask] = 0
            new_components -= 1
    if verbose:
        logging.info(
            f"Number of components removed: {n_components - new_components} -> Remaining: {new_components}"
        )

    return filtered_ccp, new_components


def post_processing(
    pred_mask,
    brain_mask_rim=None,
    sigma=0,
    dilate=False,
    min_component_size=2,
    verbose=False,
):
    # "Erosion" - Removing single voxel components
    seg_ccp, seg_components = skimage.measure.label(
        pred_mask, background=0, return_num=True, connectivity=2
    )
    seg_ccp, num_seg_components_post = remove_small_components(
        seg_ccp, size=min_component_size, verbose=verbose
    )

    # Denoising - Removing small components
    if sigma > 0.0:
        smoothed_seg_mask = skf.gaussian(seg_ccp > 0, sigma=sigma) > 0
        seg_ccp, seg_components = skimage.measure.label(
            smoothed_seg_mask, background=0, return_num=True, connectivity=1
        )

    # Ignoring brain border
    if brain_mask_rim is not None:
        border_components = np.unique(seg_ccp[brain_mask_rim.astype(bool)])

        for i in border_components:
            if i == 0:
                continue
            seg_ccp[seg_ccp == i] = 0

    post_proc_mask = seg_ccp > 0

    # End result is similar to morphological "opening" when combined with "erosion" above
    if dilate:
        post_proc_mask = skimage.morphology.dilation(post_proc_mask, footprint=ball(1))

    return post_proc_mask


def get_best_thresholds(
    anomaly_scores,
    label_masks,
    brain_masks,
    brain_mask_erode_radius=2,
    min_component_size=3,
    num_thresholds_search=100,
    start_percentile_thresh=0.90,
    stop_percentile_thresh=0.9999,
):
    seg_thresholds = []
    sample_thresh_dists = []

    eroded_brain_masks = erode_brain_masks(brain_masks, brain_mask_erode_radius)
    brain_mask_rims = (brain_masks * (~eroded_brain_masks)).astype(float)

    progress_bar = tqdm(range(len(anomaly_scores)), desc="# Processed: ?")
    for sample_idx in progress_bar:
        skull_mask = brain_mask_rims[sample_idx]
        ref_mask = post_processing(
            label_masks[sample_idx], skull_mask, min_component_size=min_component_size
        )
        y_ref = torch.from_numpy(ref_mask)
        y_ref = (
            torch.nn.functional.one_hot(y_ref.long(), 2).unsqueeze(0).permute(0, 4, 1, 2, 3)
        )
        pred = anomaly_scores[sample_idx]

        threshes = np.linspace(
            np.quantile(pred, start_percentile_thresh),
            np.quantile(pred, stop_percentile_thresh),
            num_thresholds_search,
        )

        dists = []
        threshes_loop = tqdm(threshes, desc="Searching best threshold:", leave=False)
        for t in threshes_loop:
            pred_mask = pred > t
            pred_mask = post_processing(pred_mask, brain_mask_rim=skull_mask, dilate=True)
            y_pred = torch.from_numpy(pred_mask).float().unsqueeze(0)
            y_pred = torch.nn.functional.one_hot(y_pred.long(), 2).permute(0, 4, 1, 2, 3)
            d = compute_average_surface_distance(y_ref, y_pred, symmetric=True).item()

            if len(dists) > 10 and (d - np.mean(dists[-10:])) > 1:
                break

            dists.append(d)

        dists = np.asarray(dists)
        best_thresh = threshes[dists.argmin()]
        seg_thresholds.append(best_thresh)
        sample_thresh_dists.append(dists)
        # print(f"Searched {len(dists)}, Found @ idx {dists.argmin()}: {dists[dists.argmin()]}]")
        progress_bar.set_description("# Processed: {:d}".format(sample_idx + 1))

    return seg_thresholds, sample_thresh_dists


def compute_segmentation_metrics(pred_labels, true_labels):
    metrics_df = pd.DataFrame(
        columns=[
            "TP",
            "FP",
            "FN",
            "ground_truth_components",
            "segmentation_components",
            "hausdorff",
            "mean_surf_dist",
            "dice",
            "IOL" # intersection-over-lesion 
        ]
    )

    for sample_idx in range(len(pred_labels)):
        pred = pred_labels[sample_idx]
        lab = true_labels[sample_idx]

        ref_ccp, ref_components = skimage.measure.label(
            lab, background=0, return_num=True, connectivity=3
        )
        y_ref = torch.from_numpy(lab)
        y_ref = (
            torch.nn.functional.one_hot(y_ref.long(), 2).unsqueeze(0).permute(0, 4, 1, 2, 3)
        )

        seg_ccp, _ = skimage.measure.label(
            pred, background=0, return_num=True, connectivity=3
        )
        y_pred = torch.from_numpy(pred).float().unsqueeze(0)
        y_pred = torch.nn.functional.one_hot(y_pred.long(), 2).permute(0, 4, 1, 2, 3)

        hauss_ref_to_pred = compute_hausdorff_distance(
            y_ref, y_pred, include_background=False, directed=True, percentile=99
        )
        mean_surf_dist = compute_average_surface_distance(y_ref, y_pred, symmetric=False)
        perf_dict = segmentation_metrics(ref_ccp, seg_ccp)

        perf_dict["hausdorff"] = hauss_ref_to_pred.item()
        perf_dict["mean_surf_dist"] = mean_surf_dist.item()
        
        dsc, iol = dice(pred, lab)
        perf_dict['dice'] = dsc
        perf_dict['IOL'] = iol

        metrics_df.loc[sample_idx] = perf_dict

    metrics_df["TPR"] = metrics_df.TP / metrics_df.ground_truth_components
    metrics_df["FNR"] = metrics_df.FN / metrics_df.ground_truth_components
    metrics_df["PPV"] = metrics_df.TP / (metrics_df.TP + metrics_df.FP)

    return metrics_df

"""
From wikipedia: https://en.wikipedia.org/wiki/Golden-section_search
Python program for golden section search.  This implementation
does not reuse function evaluations and assumes the minimum is c
or d (not on the edges at a or b)
"""

def gss(f, a, b, tolerance=1e-5):
    """
    Golden-section search
    to find the minimum of f on [a,b]

    * f: a strictly unimodal function on [a,b]

    Example:
    >>> def f(x): return (x - 2) ** 2
    >>> x = gss(f, 1, 5)
    >>> print(f"{x:.5f}")
    2.00000

    """
    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi

    while b - a > tolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if f(c) < f(d):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2


def auto_compute_thresholds(training_samples, false_positive_rate):

    def optimization_fn(thresh):
        fpr_at_thresh = (training_samples > thresh).mean()
        dist = np.sqrt((false_positive_rate - fpr_at_thresh) ** 2)
        return dist

    return gss(
        optimization_fn, training_samples.min(), training_samples.max(), tolerance=1e-5
    )


def dice(im1, im2):
    """
    Computes Dice score for two images

    Parameters
    ==========
    im1: Numpy Array/Matrix; Predicted segmentation in matrix form
    im2: Numpy Array/Matrix; Ground truth segmentation in matrix form

    Output
    ======
    dice_score: Dice score between two images
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (
        2.0 * (intersection.sum()) / (im1.sum() + im2.sum()),
        intersection.sum() / im2.sum(),
    )
