import os
import pickle as pk
from collections import deque
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from gravann.functions import binder as function_binder
from gravann.labels import binder as label_binder
from gravann.network import layers, encodings
from gravann.output import plot_saver
from . import training_initializer, losses, validator
from .losses import *


def run_training_configuration(cfg: Dict) -> pd.DataFrame:
    """Runs a specific parameter configuration.

    Args:
        cfg: dictionary with the configuration for the training run

    Returns:
        results
    """
    # Initialize the environment and prepare the run folder
    run_folder = training_initializer.init_environment(cfg)
    encoding = encodings.get_encoding(cfg["encoding"])
    activation = layers.get_activation_layer(cfg["activation"])
    loss_fn = getattr(losses, cfg["loss"])
    # Initialize the model and associated torch training utility
    model, early_stopper, optimizer, scheduler = training_initializer.init_model_and_optimizer(
        run_folder=run_folder,
        encoding=encoding,
        n_neurons=cfg["n_neurons"],
        activation=activation,
        model_type=cfg["model_type"],
        omega=cfg["omega"],
        hidden_layers=cfg["hidden_layers"],
        learning_rate=cfg["learning_rate"]
    )
    # Start with a pretrained model if a path is given
    if pretrained_model := cfg.get("pretrained_model", ""):
        model.load_state_dict(torch.load(pretrained_model))
        print(f"Loaded existing model from {pretrained_model}")
    # Initialize the target point sampler
    target_points_sampler = training_initializer.init_training_sampler(
        sample=cfg["sample"], target_sample_method=cfg["sample_method"],
        sample_domain=cfg["sample_domain"], batch_size=cfg["batch_size"]
    )
    # Initialize the prediction function by binding the model and defined configuration parameters
    prediction_fn = function_binder.bind_integration(
        method=cfg["integrator"],
        use_acc=cfg["use_acceleration"],
        model=model,
        encoding=encoding,
        integration_points=cfg["integration_points"],
        integration_domain=cfg["integration_domain"],
    )
    # Initialize the label function by binding the sample data
    label_fn = label_binder.bind_label(
        method=cfg["ground_truth"],
        use_acc=cfg["use_acceleration"],
        sample=cfg["sample"],
        resolution=cfg["resolution"]
    )
    # Add noise on top (if defined)
    label_fn = function_binder.bind_noise(
        cfg["noise"][0],
        label_fn,
        *cfg["noise"][1:]
    )

    # When a new network is created: init empty training logs and loss trend indicators
    loss_log, lr_log, weighted_average_log, n_inferences = [], [], [], []
    weighted_average = deque([], maxlen=20)
    target_points, labels = [], []

    t = tqdm(range(cfg["iterations"]), ncols=150)
    # At the beginning (first plots) we assume no learned c
    c = 1.
    for it in t:
        # Each ten epochs we resample the target points
        if it % 10 == 0:
            target_points = target_points_sampler()
            labels = label_fn(target_points)

        # Train
        loss, c = _train_on_batch(target_points, prediction_fn, labels, loss_fn, optimizer, scheduler)

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        lr_log.append(optimizer.param_groups[0]['lr'])
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        n_inferences.append((cfg["integration_points"] * cfg["batch_size"]) // 1000)
        wa_out = np.mean(weighted_average)

        t.set_postfix_str(f"L={loss.item():.3e} | AvgL={wa_out:.3e} | c={c:.3e}")

        if early_stopper.early_stop(loss.item(), model):
            print(f"Early stopping at minimal loss {early_stopper.minimal_loss}")
            break

        torch.cuda.empty_cache()
    # Restore best checkpoint
    print("Restoring best checkpoint for validation...")
    model.load_state_dict(torch.load(os.path.join(run_folder, "best_model.mdl")))

    print("Saving the config file...")
    with open(os.path.join(run_folder, 'config.pk'), 'wb') as handle:
        pk.dump(cfg, handle)

    # Compute the validation
    print("Validating...")
    validation_kwargs = {
        "N": cfg["validation_points"],
        "progressbar": False,
        "sampling_altitudes": cfg["validation_sampling_altitudes"],
        "integration_points": cfg["integration_points"],
        "batch_size": cfg["validation_batch_size"]
    }
    validation_results = validator.validate(
        model,
        encoding,
        cfg["sample"],
        cfg["validation_ground_truth"],
        cfg.get("use_acceleration", True),
        **validation_kwargs
    )

    print("Saving...")
    plot_saver.save_results(loss_log, weighted_average_log, validation_results, model, run_folder)

    plot_saver.save_plots_v2(
        model, encoding, cfg["sample"],
        lr_log, loss_log, weighted_average_log,
        n_inferences, run_folder, c, cfg["plotting_points"]
    )

    # Compute validation results
    val_res = validator.validation_results_unpack_df(validation_results)

    results_df = pd.concat([pd.DataFrame([cfg]), val_res], axis=1)
    return results_df


def _train_on_batch(points, prediction_fn, labels, loss_fn, optimizer, scheduler):
    """Trains the passed model on the passed batch

    Args:
        points (tensor): target points for training
        prediction_fn (func): prediction func of the model
        labels (tensor): labels at the target points
        loss_fn (func): loss function for training
        optimizer (torch optimizer): torch optimizer to use
        scheduler (torch LR scheduler): torch LR scheduler to use

    Returns:
        torch tensor: losses
    """
    predicted = prediction_fn(points)
    c = torch.sum(predicted * labels) / torch.sum(predicted * predicted)

    if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss]:
        loss = loss_fn(predicted, labels)
    else:
        loss = loss_fn(predicted.view(-1), labels.view(-1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step(loss.item())

    return loss, c
