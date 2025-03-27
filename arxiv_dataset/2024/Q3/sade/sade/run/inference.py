"""Streamlined inference"""

import glob
import json
import logging
import os
import re

import models.registry as registry
import numpy as np
import torch
from datasets.loaders import get_dataloaders
from tqdm import tqdm

from sade.models.ema import ExponentialMovingAverage
from sade.ood_detection_helper import auxiliary_model_analysis

from .utils import restore_pretrained_weights


def inference(config, workdir):
    """Runs a full inference pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    assert (
        config.eval.checkpoint_num > 0
    ), "Please specify a checkpoint number using `config.eval.checkpoint_num`"
    assert os.path.exists(
        config.eval.experiment.flow_checkpoint_path
    ), "Please provide a checkpoint for the flow model"

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
    checkpoint_num = config.eval.checkpoint_num
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {os.path.abspath(checkpoint_path)}"
        )

    state = restore_pretrained_weights(checkpoint_path, state, config.device)
    checkpoint_step = state["model_checkpoint_step"]
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(
        config, score_model, return_norm=False, denoise=config.msma.denoise
    )
    score_normer = registry.get_msma_score_fn(
        config, score_model, return_norm=True, denoise=config.msma.denoise
    )
    logging.info(f"Loaded score model from checkpoint {checkpoint_step:d}...")

    # Initialize flow model
    flow_path = os.path.abspath(config.eval.experiment.flow_checkpoint_path)
    ckpt_path = f"{flow_path}/checkpoint.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{flow_path}/checkpoint-meta.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Could not find flow checkpoint or checkpoint-meta at {flow_path}"
        )
    else:
        logging.info(f"Found flow checkpoint at {ckpt_path}")

    flownet = registry.create_flow(config).eval().requires_grad_(False)
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    _ = state_dict["model_state_dict"].pop("position_encoder.cached_penc", None)
    flownet.load_state_dict(state_dict["model_state_dict"], strict=True)
    logging.info(
        f"Loaded flow model at iter={state_dict['kimg']}, val_loss={state_dict['val_loss']}"
    )
    # Use user-specified chunk size for evaluation
    flownet.patch_batch_size = config.flow.patch_batch_size

    experiment = config.eval.experiment
    if experiment.id == "default":
        experiment.id += f"-ckpt-{checkpoint_num}"
    save_dir = os.path.join(workdir, "experiments", experiment.id)
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Running experiment {experiment.id}")

    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config.to_dict(), fp=f)

    model_states = {}
    model_states["score_model_checkpoint_step"] = state["model_checkpoint_step"]
    model_states["flow_model_kimg"] = state_dict["kimg"]
    model_states["flow_model_vakl_loss"] = state_dict["val_loss"]

    with open(f"{save_dir}/model_states.json", "w") as f:
        json.dump(model_states, fp=f)

    enhance_lesions = False
    if "-enhanced" in experiment.ood:
        enhance_lesions = True
        experiment.ood = experiment.ood.split("-")[0]
    
    dsnames = [experiment.train, experiment.inlier, experiment.ood]
    _, datasets = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
        num_workers=4,
        infinite_sampler=False,
    )



    for name, ds in zip(dsnames, datasets):

        if 'lesion' in name and enhance_lesions:
            name += '-enhanced'

        os.makedirs(f"{save_dir}/{name}/", exist_ok=True)

        for i, xdict in enumerate(tqdm(ds)):
            fname = os.path.basename(ds.data[i]["image"])
            sampleid = fname.split(".")[0]
            x = xdict["image"].to(config.device).unsqueeze(0)

            if "lesion" in name and enhance_lesions:
                labels = xdict["label"].to(config.device)
                x = x * labels * 1.5 + x * (1 - labels)

            scores = scorer(x)
            heatmap = -flownet.log_density(scores, x, fast=True)
            background_mask = (x > x.min()).sum(1)
            heatmap = heatmap * background_mask

            score_norms = score_normer(x).cpu().numpy()
            heatmap = heatmap.squeeze().cpu().numpy()
            scores = scores.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            save_dict = dict(
                original=x,
                heatmap=heatmap,
                scores=scores,
                score_norms=score_norms,
            )
            if "label" in xdict:
                save_dict["segmentation"] = xdict["label"].cpu().squeeze(0)

            np.savez_compressed(
                f"{save_dir}/{name}/{sampleid}.npz",
                **save_dict
            )
            torch.cuda.empty_cache()
