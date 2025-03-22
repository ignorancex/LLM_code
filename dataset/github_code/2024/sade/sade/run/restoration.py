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


def get_pc_restorer(
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
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())

    # Create predictor & corrector update functions
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

    @torch.no_grad()
    def pc_restorer(model, x_batch, num_forward_steps):
        """The PC restorer funciton that does a forward then backward step.

        Args:
          model: A score model.
          x: A mini-batch of data.
          num_forward_steps: An integer. The number of forward SDE steps for perturbing the data.
        Returns:
          Samples, number of function evaluations.
        """

        timesteps = torch.linspace(sde.T, eps, sde.N, device=x_batch.device)
        start_idx = sde.N - num_forward_steps
        start_t = timesteps[start_idx]

        z = torch.randn_like(x_batch)
        mean, std = sde.marginal_prob(x_batch, t=start_t)
        x_init = mean + sde._unsqueeze(std) * z
        x = x_init

        for i in tqdm(range(start_idx, sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(x.shape[0], device=t.device) * t
            # pdb.set_trace()
            x, x_mean = corrector_update_fn(x, vec_t, model=model)
            x, x_mean = predictor_update_fn(x, vec_t, model=model)

        return x_mean if denoise else x

    return pc_restorer


def restorer(config, workdir, num_evals=10):
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
    save_dir = os.path.join(workdir, "eval", "restoration")
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving inpainting results to {save_dir}")
    experiment = config.eval.experiment
    experiment_name = f"{experiment.inlier}_{experiment.ood}"
    enhance_lesions = False
    if "-enhanced" in experiment.ood:
        enhance_lesions = True
        experiment.ood = experiment.ood.split("-")[0]
    logging.info(f"Running epxperiment {experiment_name}")

    # Load datasets
    dataloaders, _ = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
        num_workers=4,
        infinite_sampler=False,
    )

    _, inlier_ds, ood_ds = dataloaders

    # Initialize sampling routines
    sde = registry.create_sde(config)
    restoration_fn = get_pc_restorer(
        sde,
        predictor=config.sampling.predictor,
        corrector=config.sampling.corrector,
        snr=config.sampling.snr,
    )
    num_forward_steps = sde.N // 4  # Following AnoDDPM

    x_inlier_results = {}
    x_ood_results = {}

    for result_dict, ds in zip([x_inlier_results, x_ood_results], [inlier_ds, ood_ds]):
        result_dict["imputed_images"] = []
        result_dict["errors"] = []

        for x_img_dict in tqdm(ds):
            x = x_img_dict["image"].to(config.device)

            if enhance_lesions and "label" in x_img_dict:
                labels = x_img_dict["label"].to(config.device)
                x = x * labels * 1.5 + x * (1 - labels)

            x_restored = torch.zeros_like(x)
            for _ in range(num_evals):
                x_restored += restoration_fn(score_model, x, num_forward_steps)
            x_restored /= num_evals
            x_error = torch.abs(x_restored - x).sum(1)
            result_dict["errors"].append(x_error.cpu().numpy())
            result_dict["imputed_images"].append(x_restored.cpu().numpy())

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


if __name__ == "__main__":
    from sade.configs.ve import biggan_config

    workdir = sys.argv[1]

    config = biggan_config.get_config()
    config.fp16 = True
    config.eval.batch_size = 16
    experiment = config.eval.experiment
    experiment.train = "abcd-val"  # The dataset used for training MSMA
    experiment.inlier = "abcd-test"
    experiment.ood = "lesion_load_20-enhanced"

    restorer(config, workdir)
