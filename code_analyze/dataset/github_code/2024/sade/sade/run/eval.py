import glob
import logging
import os
import re

import ants
import models.registry as registry
import numpy as np
import torch
from datasets.loaders import get_dataloaders
from tqdm import tqdm

from sade.metrics import (
    auto_compute_thresholds,
    compute_segmentation_metrics,
    erode_brain_masks,
    get_best_thresholds,
    post_processing,
)
from sade.models.ema import ExponentialMovingAverage
from sade.ood_detection_helper import auxiliary_model_analysis

from .utils import restore_pretrained_weights


def evaluator(config, workdir):
    """Runs the evaluation pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Initialize model.
    score_model = registry.create_model(config, print_summary=True)

    # Load checkpoint
    state = dict(
        model=score_model,
        step=0,
        ema=ExponentialMovingAverage(score_model.parameters(), decay=0),
        model_checkpoint_step=0,
    )
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    if config.eval.checkpoint_num > 0:
        checkpoint_num = config.eval.checkpoint_num
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num}.pth")
    else:
        # Get the latest checkpoint
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
        if len(checkpoint_paths) > 0:
            checkpoint_path = max(
                checkpoint_paths, key=lambda x: int(re.search(r"_(\d+)\.pth", x).group(1))
            )
        else:  # Try to check for checkpoint-meta
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-meta.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Could not find checkpoint or checkpoint-meta at {os.path.dirname(checkpoint_path)}"
                )

    state = restore_pretrained_weights(checkpoint_path, state, config.device)
    checkpoint_step = state["model_checkpoint_step"]
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(
        config, score_model, return_norm=config.msma.l2_normed, denoise=config.msma.denoise
    )
    logging.info(f"Evaluating model at checkpoint {checkpoint_step:d}...")

    # Create save directory
    save_dir = os.path.join(
        workdir,
        "eval",
        f"ckpt_{checkpoint_step}",
        f"smin={config.msma.min_timestep:.2f}_smax={config.msma.max_timestep:.2f}_t={config.msma.n_timesteps}",
    )
    experiment = config.eval.experiment
    experiment_name = f"{experiment.train}_{experiment.inlier}_{experiment.ood}"
    experiment_name += "-denoise" if config.msma.denoise else ""
    experiment_name += "-raw" if not config.msma.l2_normed else ""
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving experiment {experiment_name} results to {save_dir}")

    # Build data iterators
    if "-enhanced" in experiment.ood:
        experiment.ood = experiment.ood.split("-")[0]

    test_dataloaders, datasets = get_dataloaders(
        config,
        evaluation=True,
        num_workers=4,
        infinite_sampler=False,
    )

    eval_dl, inlier_dl, ood_dl = test_dataloaders

    # Run score norm evaluation
    logging.info(f"Running experiment: {experiment_name}")
    x_eval_scores = []
    x_inlier_scores = []
    x_ood_scores = []

    for res_arr, ds in zip(
        [x_eval_scores, x_inlier_scores, x_ood_scores], [eval_dl, inlier_dl, ood_dl]
    ):
        ds_iter = iter(ds)
        for batch in tqdm(ds_iter):
            x = batch["image"].to(config.device)
            scores = scorer(x).cpu()
            res_arr.append(scores)

    x_eval_scores = torch.cat(x_eval_scores).numpy()
    x_inlier_scores = torch.cat(x_inlier_scores).numpy()
    x_ood_scores = torch.cat(x_ood_scores).numpy()

    # Run ood detection

    results = auxiliary_model_analysis(
        x_eval_scores,
        x_inlier_scores,
        [x_ood_scores],
        components_range=range(3, 6, 1),
        labels=["Train", "Inlier", "OOD"],
        verbose=0,
    )

    experiment_dict = {
        "eval_score_norms": x_eval_scores,
        "inlier_score_norms": x_inlier_scores,
        "ood_score_norms": x_ood_scores,
        "results": results,
    }

    np.savez_compressed(
        f"{save_dir}/{experiment_name}_results.npz",
        **experiment_dict,
    )

    # Print results
    print(results["GMM"]["metrics"])


def segmentation_evaluator_v1(config, workdir):
    config.device = torch.device("cpu")
    experiment = config.eval.experiment
    experiment_name = f"{experiment.inlier}_{experiment.ood}"
    scores_path = f"{workdir}/{experiment_name}_results.npz"
    assert os.path.exists(scores_path), f"Scores not found at {scores_path}"
    data = np.load(scores_path, allow_pickle=True)
    x_ood_scores = data["ood"]

    if "-enhanced" in experiment.ood:
        experiment.ood = experiment.ood.split("-")[0]

    (_, inlier_ds, ood_ds), _ = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
    )

    x_ood = []
    x_ood_labels = []
    for x in ood_ds:
        x_ood.append(x["image"])
        x_ood_labels.append(x["label"])

    x_ood = torch.cat(x_ood)
    x_ood_labels = torch.cat(x_ood_labels)
    ood_brain_masks = (x_ood != -1.0).sum(dim=1).bool()

    ### Run the segmentation evaluation pipeline

    # Ensure same spatial size
    anomaly_scores = (
        torch.nn.functional.interpolate(
            torch.from_numpy(x_ood_scores).unsqueeze(1), size=x_ood.shape[2:]
        )
        .squeeze(1)
        .numpy()
    )
    anomaly_scores = anomaly_scores - anomaly_scores.min(axis=(1, 2, 3), keepdims=True)
    anomaly_scores = anomaly_scores * ood_brain_masks

    true_labels = x_ood_labels.sum(1).cpu().numpy() > 0
    true_labels_masked = true_labels * ood_brain_masks

    # Computing Hausdorff distance to determine best thresholds for segmentation
    best_seg_thresholds, _ = get_best_thresholds(
        anomaly_scores, true_labels_masked, ood_brain_masks, min_component_size=3
    )

    post_proc_preds = []
    post_proc_labels = []

    eroded_brain_masks = erode_brain_masks(ood_brain_masks)
    brain_mask_rims = (ood_brain_masks * (~eroded_brain_masks)).astype(float)
    for sample_idx, thresh in enumerate(best_seg_thresholds):
        skull_mask = brain_mask_rims[sample_idx]
        pred_scores = anomaly_scores[sample_idx]
        best_thresh = best_seg_thresholds[sample_idx]
        pred = post_processing(
            pred_scores > best_thresh, skull_mask, dilate=True, min_component_size=3
        )
        ref_mask = post_processing(
            true_labels[sample_idx], skull_mask, min_component_size=3
        )
        post_proc_preds.append(pred)
        post_proc_labels.append(ref_mask)

    # Save the predictions
    post_proc_preds = np.stack(post_proc_preds)
    post_proc_labels = np.stack(post_proc_labels)
    np.savez_compressed(
        f"{workdir}/{experiment_name}_segs.npz",
        **{"preds": post_proc_preds, "labels": post_proc_labels},
    )

    metrics_df = compute_segmentation_metrics(post_proc_preds, post_proc_labels)
    metrics_df.to_csv(f"{workdir}/{experiment_name}_seg_eval.csv")
    print(metrics_df.dropna().describe())


def segmentation_evaluator(config, workdir):

    config.training.batch_size = 1
    config.eval.batch_size = 1
    config.data.cache_rate = 0.0
    experiment = config.eval.experiment
    dataset_name = experiment.ood
    expid = experiment.id

    workdir = f"{workdir}/experiments/{expid}"
    assert os.path.exists(
        workdir
    ), f"Could not find experiment directory: {workdir}.\
        Please make sure the heatmaps are pre-computed in experiment directory."

    if "-enhanced" in experiment.ood:
        experiment.ood = experiment.ood.split("-")[0]

    _, datasets = get_dataloaders(
        config, evaluation=True, num_workers=1, infinite_sampler=False
    )
    *_, ood_ds = datasets

    fpaths = glob.glob(f"{workdir}/{dataset_name}/*.npz")
    post_proc_preds_at_fpr10 = []
    post_proc_preds_at_fpr05 = []
    post_proc_labels = []

    # Compute segmentation predicitons
    if not os.path.exists(f"{workdir}/autothresholds.npy"):
        print("Computing automatic thresholds ...")
        compute_auto_thresholds(config, workdir)

    autothresh = np.load(f"{workdir}/autothresholds.npy", allow_pickle=True).item()

    for i, fname in enumerate(tqdm(fpaths)):
        data = np.load(fname)
        #     x = data['original'] + 1
        xdict = ood_ds[i]
        x = xdict["image"]
        brain_mask = (x > x.min()).sum(axis=0).bool().numpy()
        brain_mask = ants.from_numpy(brain_mask).astype("float32")
        eroded_brain_mask = (
            ants.morphology(
                brain_mask,
                operation="erode",
                radius=2,
                mtype="binary",
                shape="ball",
                value=1,
                radius_is_parametric=True,
            )
            .numpy()
            .astype(float)
        )

        #     gt = (data['segmentation'] > 0) * eroded_brain_mask
        gt = (xdict["label"][0]) * eroded_brain_mask
        gt = post_processing(gt, min_component_size=3)
        post_proc_labels.append(gt)

        pred_scores = data["heatmap"]
        pred_scores = (pred_scores - pred_scores.min()) * eroded_brain_mask

        pred = post_processing(
            pred_scores > autothresh["thresh_fpr10"], dilate=True, min_component_size=3
        )
        post_proc_preds_at_fpr10.append(pred)

        pred = post_processing(
            pred_scores > autothresh["thresh_fpr05"], dilate=True, min_component_size=3
        )
        post_proc_preds_at_fpr05.append(pred)

    # Save the predictions
    post_proc_preds_at_fpr05 = np.stack(post_proc_preds_at_fpr05)
    post_proc_preds_at_fpr10 = np.stack(post_proc_preds_at_fpr10)
    post_proc_labels = np.stack(post_proc_labels)
    np.savez_compressed(
        f"{workdir}/{dataset_name}_segs.npz",
        **{
            "preds_fpr05": post_proc_preds_at_fpr05,
            "preds_fpr10": post_proc_preds_at_fpr10,
            "labels": post_proc_labels,
        },
    )

    # Compute performance metrics
    for fpr, preds in zip(
        ["fpr05", "fpr10"], [post_proc_preds_at_fpr05, post_proc_preds_at_fpr10]
    ):
        experiment_name = f"{dataset_name}-{fpr}"
        print(experiment_name)
        metrics_df = compute_segmentation_metrics(preds, post_proc_labels)
        metrics_df.to_csv(f"{workdir}/{experiment_name}_seg_eval.csv")
        print(metrics_df.dropna().describe())


def compute_auto_thresholds(config, experiment_dir):
    exp = config.eval.experiment
    inlier_ds = [exp.train, exp.inlier]
    savedir = f"{experiment_dir}/inliers-flat-post-proc/"
    os.makedirs(savedir, exist_ok=True)

    # Compute inlier heatmaps
    for ds in inlier_ds:
        dirname = f"{experiment_dir}/{ds}"
        samples = glob.glob(f"{dirname}/*.npz")
        for fname in tqdm(samples):
            with np.load(fname) as data:
                sid = fname.split("/")[-1].split(".npz")[0]
                heatmap = data["heatmap"]
                ximg = ants.from_numpy(data["original"][0] + 1) / 2
                brain_mask = (ximg > 0).astype("float32")
                eroded_mask = (
                    ants.morphology(
                        brain_mask,
                        operation="erode",
                        radius=2,
                        mtype="binary",
                        shape="ball",
                        value=1,
                        radius_is_parametric=True,
                    )
                    .numpy()
                    .astype(bool)
                )
                anomaly_scores = (heatmap - heatmap.min()) * eroded_mask
                x = anomaly_scores[anomaly_scores.nonzero()]
                np.savez_compressed(f"{savedir}/{sid}.npz", x)

    # Load inlier samples and compute thresholds
    samples = glob.glob(f"{savedir}/*")
    arr = []
    for fname in tqdm(samples):
        with np.load(fname) as data:
            arr.append(data["arr_0"])

    arr = np.concatenate(arr).reshape(-1)

    thresh_fpr10 = auto_compute_thresholds(training_samples=arr, false_positive_rate=0.1)
    thresh_fpr05 = auto_compute_thresholds(training_samples=arr, false_positive_rate=0.05)
    np.save(
        f"{experiment_dir}/autothresholds.npy",
        {"thresh_fpr05": thresh_fpr05, "thresh_fpr10": thresh_fpr10},
    )
