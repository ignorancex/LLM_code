import functools
import logging
import os
import sys

import numpy as np
import torch
from sampling import (
    get_corrector,
    get_predictor,
    shared_corrector_update_fn,
    shared_predictor_update_fn,
)
from tqdm import tqdm

from sade.datasets.loaders import get_dataloaders
from sade.models import registry
from sade.models.ema import ExponentialMovingAverage
from sade.run.utils import restore_pretrained_weights

"""
Adapted from original Score SDE code:
https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/controllable_generation.py#L8C12-L8C12
"""


def get_pc_inpainter(
    sde,
    predictor,
    corrector,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=True,
    denoise=True,
    eps=1e-3,
):
    """Create an image inpainting function that uses PC samplers.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for the corrector.
      n_steps: An integer. The number of corrector steps per update of the corrector.
      probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
      continuous: `True` indicates that the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

    Returns:
      An inpainting function.
    """

    predictor = get_predictor(predictor)
    corrector = get_corrector(corrector)

    # Define predictor & corrector
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def get_inpaint_update_fn(update_fn):
        """Modify the update function of predictor & corrector to incorporate data information."""

        def inpaint_update_fn(model, data, mask, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                masked_data_mean, std = sde.marginal_prob(data, vec_t)
                masked_data = masked_data_mean + torch.randn_like(x) * sde._unsqueeze(std)
                x = x * (1.0 - mask) + masked_data * mask
                x_mean = x * (1.0 - mask) + masked_data_mean * mask
                return x, x_mean

        return inpaint_update_fn

    projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
    corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

    def pc_inpainter(model, data, mask):
        """Predictor-Corrector (PC) sampler for image inpainting.

        Args:
          model: A score model.
          data: A PyTorch tensor that represents a mini-batch of images to inpaint.
          mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
            and value `0` marks pixels that require inpainting.

        Returns:
          Inpainted (complete) images.
        """
        with torch.no_grad():
            # Initial sample
            x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1.0 - mask)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(0, sde.N, 2)):
                t = timesteps[i]
                x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t)
                x, x_mean = projector_inpaint_update_fn(model, data, mask, x, t)

            return x_mean if denoise else x

    return pc_inpainter


def checkerboard_mask(image_size, patch_size, inverted=False):
    channels, height, width, depth = image_size
    mask = torch.zeros(image_size)
    flag = inverted

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            for k in range(0, depth, patch_size):
                if flag:
                    mask[:, i : i + patch_size, j : j + patch_size, k : k + patch_size] = 1
                flag = not flag
    return mask


def inpainter(config, workdir, patch_size=3, num_evals=10):
    # Initialize score model
    score_model = registry.create_model(config, log_grads=False)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0, model_checkpoint_step=0)

    # Get the score model checkpoint from pretrained run
    state = restore_pretrained_weights(
        config.training.pretrained_checkpoint, state, config.device
    )
    score_model.eval().requires_grad_(False)

    # Create save directory
    save_dir = os.path.join(workdir, "eval", "inpainting")
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving inpainting results to {save_dir}")
    experiment = config.eval.experiment
    experiment_name = f"{experiment.inlier}_{experiment.ood}"
    enhance_lesions = False
    if "-enhanced" in experiment.ood:
        enhance_lesions = True
        experiment.ood = experiment.ood.split("-")[0]
    logging.info(f"Running experiment {experiment_name}")

    # Load datasets
    dataloaders, _ = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
        num_workers=4,
        infinite_sampler=False,
    )

    _, inlier_ds, ood_ds = dataloaders

    # Initialize inpainting routines
    sde = registry.create_sde(config)
    inpainting_fn = get_pc_inpainter(
        sde,
        predictor=config.sampling.predictor,
        corrector=config.sampling.corrector,
        snr=config.sampling.snr,
    )

    # Number of sampling evaluations to average over
    # Loosely inspired by LMD
    shape = (config.data.num_channels, *config.data.image_size)
    CHECKERBOARD_MASK = checkerboard_mask(shape, patch_size=patch_size)
    CHECKERBOARD_MASK = CHECKERBOARD_MASK.unsqueeze(0).to(config.device)

    x_inlier_results = {}
    x_ood_results = {}

    for result_dict, ds in zip([x_inlier_results, x_ood_results], [inlier_ds, ood_ds]):
        result_dict["errors"] = []
        result_dict["imputed_images"] = []

        for x_img_dict in tqdm(ds):
            x = x_img_dict["image"].to(config.device)

            if enhance_lesions and "label" in x_img_dict:
                labels = x_img_dict["label"].to(config.device)
                x = x * labels * 1.5 + x * (1 - labels)

            brain_mask = (x != -1).sum(1, keepdims=True) > 0
            # Produce a 3D mask with checkerboard pattern of a given size
            # m = get_checkerboard_mask(x.shape).to(config.device)
            m = CHECKERBOARD_MASK.repeat(x.shape[0], 1, 1, 1, 1)
            x_inpainted = torch.zeros_like(x)
            for _ in range(num_evals):
                x_inpainted += inpainting_fn(score_model, x, brain_mask * m)
                # Invert each round
                m = 1 - m
            x_inpainted /= num_evals
            x_error = torch.abs(x_inpainted - x).sum(1)
            result_dict["errors"].append(x_error.cpu().numpy())
            result_dict["imputed_images"].append(x_inpainted.cpu().numpy())

    x_inlier_errors = np.concatenate(x_inlier_results["errors"])
    x_ood_errors = np.concatenate(x_ood_results["errors"])
    x_inlier_imputed = np.concatenate(x_inlier_results["imputed_images"])
    x_ood_imputed = np.concatenate(x_ood_results["imputed_images"])

    np.savez_compressed(
        f"{save_dir}/{experiment_name}_results.npz",
        **{
            "inliers": x_inlier_errors,
            "ood": x_ood_errors,
            "inlier_imputed": x_inlier_imputed,
            "ood_imputed": x_ood_imputed,
        },
    )

    return


if __name__ == "__main__":
    from sade.configs.ve import biggan_config

    logging.basicConfig(level=logging.INFO)

    workdir = sys.argv[1]

    config = biggan_config.get_config()
    config.fp16 = True
    config.eval.batch_size = 1
    experiment = config.eval.experiment
    experiment.train = "abcd-val"  # The dataset used for training MSMA
    experiment.inlier = "abcd-test"
    experiment.ood = "lesion_load_20-enhanced"

    inpainter(config, workdir)
