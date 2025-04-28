import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gravann.functions import binder as function_binder
from gravann.labels import binder as label_binder
from gravann.util import fixRandomSeeds
from gravann.util import get_target_point_sampler
from .losses import contrastive_loss, normalized_L1_loss, normalized_relative_L2_loss, \
    normalized_relative_component_loss, RMSE, relRMSE


def validate(model, encoding, sample, ground_truth, use_acc=True, **kwargs):
    """Convenience function to compute the different loss values for the passed model and asteroid with high precision

    Args:
        model: trained model
        encoding (encoding): encoding to use for the points
        sample (str): the name of the asteroid (also used as filepath)
        ground_truth (str): 'polyhedral' or 'mascon' or 'polyhedral-mascon' (polyhedral as groundtruth to mascon model)
        use_acc (bool, optional): if to use the acceleration labels instead of the potential
        **kwargs: see below

    Keyword Args:
        integration_points (int, optional): Number of integrations points to use. Defaults to 500000.
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        mascon_masses_nu (torch.tensor): non-uniform asteroid masses. Pass if using differential training
        mesh_vertices ((N, 3) array): mesh vertices
        mesh_faces ((M, 3) array): mesh triangles
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        batch_size (int, optional): batch size (will split N in batches). Defaults to 100.
        russell_points (int , optional): how many points should be sampled per altitude for russel style radial projection sampling. Defaults to 3.
        progressbar (bool, optional): Display a progress. Defaults to True.
        noise (list) if given then the noise is added on the ground truth for validation

    Returns:
        pandas dataframe: Results as df

    """
    label_function = label_binder.bind_label(ground_truth, use_acc, sample=sample, **kwargs)
    if noise_cfg := kwargs.get("noise", None):
        label_function = function_binder.bind_noise(noise_cfg[0], label_function, noise_cfg[1:])
    prediction_function = function_binder.bind_integration('trapezoid', use_acc, model, encoding, **kwargs)
    return _validation(label_function, prediction_function, sample, **kwargs)


def _validation(label_function, prediction_function, sample, **kwargs):
    """Computes different loss values for the passed model and asteroid with high precision

    Args:
        label_function: the original training labels
        prediction_function: the prediction function of the trained model
        sample (str): name of the body (equals file name)

    Keyword Args:
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        batch_size (int, optional): batch size (will split N in batches). Defaults to 32.
        russell_points (int , optional): how many points should be sampled per altitude for russel style radial projection sampling. Defaults to 3.
        progressbar (bool, optional): Display a progress. Defaults to True.

    Returns:
        pandas dataframe: Results as df
    """

    # Default arguments for the Keyword Args
    N = kwargs.get("N", 5000)
    sampling_altitudes = kwargs.get("sampling_altitudes", [0.05, 0.1, 0.25])
    batch_size = kwargs.get("batch_size", 100)
    russell_points = kwargs.get("russell_points", 3)
    progressbar = kwargs.get("progressbar", True)

    torch.cuda.empty_cache()
    fixRandomSeeds()

    asteroid_pk_path = f"./3dmeshes/{sample}.pk"

    loss_fns = [normalized_L1_loss, normalized_relative_component_loss, RMSE, relRMSE]
    cols = ["Altitude", "Normalized L1 Loss", "Normalized Relative Component Loss", "RMSE", "relRMSE"]
    results = pd.DataFrame(columns=cols)

    ###############################################
    # Compute validation for radially projected points (outside the asteroid),

    # Low altitude
    torch.cuda.empty_cache()
    pred, labels, loss_values = [], [], []
    target_sampler = get_target_point_sampler(russell_points, method="radial_projection", bounds=[
        0.0, 0.15625], limit_shape_to_asteroid=asteroid_pk_path)

    target_points = target_sampler().detach()

    if progressbar:
        pbar = tqdm(desc="Computing validation...",
                    total=2 * len(target_points) + N * (len(sampling_altitudes)))

    for idx in range((len(target_points) // batch_size) + 1):
        indices = list(range(idx * batch_size,
                             np.minimum((idx + 1) * batch_size, len(target_points))))
        points = target_points[indices]
        labels.append(label_function(points).detach())
        prediction = prediction_function(points).detach()
        pred.append(prediction)
        if progressbar:
            pbar.update(batch_size)

    pred = torch.cat(pred)
    labels = torch.cat(labels)

    # Compute Losses
    for loss_fn in loss_fns:
        if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss, RMSE,
                       relRMSE]:
            loss_values.append(torch.mean(
                loss_fn(pred, labels)).cpu().detach().item())
        else:
            loss_values.append(torch.mean(
                loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

    results = results.append(
        dict(zip(cols, ["Low Altitude"] + loss_values)), ignore_index=True)

    # High altitude
    torch.cuda.empty_cache()
    pred, labels, loss_values = [], [], []
    target_sampler = get_target_point_sampler(russell_points, method="radial_projection", bounds=[
        0.15625, 0.3125], limit_shape_to_asteroid=asteroid_pk_path)

    target_points = target_sampler().detach()
    for idx in range((len(target_points) // batch_size) + 1):
        indices = list(range(idx * batch_size,
                             np.minimum((idx + 1) * batch_size, len(target_points))))
        points = target_points[indices]
        labels.append(label_function(points).detach())
        prediction = prediction_function(points).detach()
        pred.append(prediction)
        if progressbar:
            pbar.update(batch_size)

    pred = torch.cat(pred)
    labels = torch.cat(labels)

    # Compute Losses
    for loss_fn in loss_fns:
        if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss, RMSE,
                       relRMSE]:
            loss_values.append(torch.mean(
                loss_fn(pred, labels)).cpu().detach().item())
        else:
            loss_values.append(torch.mean(
                loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

    results = results.append(
        dict(zip(cols, ["High Altitude"] + loss_values)), ignore_index=True)

    ################################################
    # Compute errors at different altitudes
    for idx, altitude in enumerate(sampling_altitudes):
        torch.cuda.empty_cache()
        pred, labels, loss_values = [], [], []
        target_sampler = get_target_point_sampler(
            N=batch_size, method="altitude",
            bounds=[altitude], limit_shape_to_asteroid=asteroid_pk_path)
        for batch in range(N // batch_size):
            target_points = target_sampler().detach()
            labels.append(label_function(target_points).detach())

            prediction = prediction_function(target_points).detach()
            pred.append(prediction)

            if progressbar:
                pbar.update(batch_size)
            torch.cuda.empty_cache()
        pred = torch.cat(pred)
        labels = torch.cat(labels)

        # Compute Losses
        for loss_fn in loss_fns:
            if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss, RMSE,
                           relRMSE]:
                loss_values.append(torch.mean(
                    loss_fn(pred, labels)).cpu().detach().item())
            else:
                loss_values.append(torch.mean(
                    loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

        results = results.append(
            dict(zip(cols, ["Altitude_" + str(idx)] + loss_values)), ignore_index=True)

    if progressbar:
        pbar.close()

    torch.cuda.empty_cache()
    return results


def validation_results_unpack_df(validation_results):
    """Converts validation df to data row

    Args:
        validation_results (pandas.df): validation results

    Returns:
        pandas.df: df as one row
    """
    v = validation_results.set_index("Altitude")
    v = v.unstack().to_frame().sort_index(level=1).T
    v.columns = [x + '@' + str(y) for (x, y) in v.columns]
    return v
