import os
import pathlib

import numpy as np
import torch

from gravann.network import network_initalizer
from gravann.util import EarlyStopping, get_target_point_sampler


def get_run_folder(cfg: dict, create_folder: bool = False) -> str:
    """Returns a run folder path as string for a given configuration.
    Args:
        cfg: dictionary with the configuration
        create_folder: if the run_folder should also be created for the system

    Returns:
        path as string

    """
    run_folder = os.path.join(
        f"{cfg['output_folder']}",
        f"{cfg['sample']}",
        f"{cfg['ground_truth']}",
        f"it-id-{cfg['run_id']:04d}"
    )

    if create_folder:
        pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)

    return run_folder


def init_cuda_environment(cfg: dict) -> None:
    """Empties the cuda cache and inits random seed generators of numpy and torch.

    Args:
        cfg: parameters, should contain an entry {'seed': _}

    Returns:
        None

    """
    torch.cuda.empty_cache()
    # Fix the random seeds for this run
    _fixate_seeds(cfg['seed'])


def _fixate_seeds(seed: int = 42):
    """This function sets the random seeds in torch and numpy to enable reproducible behavior.

    Args:
        seed (optional): the chosen seed, defaults to 42
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def init_environment(parameters: dict) -> str:
    """Creates the environment (the run-folder) for the given training with parameters

    Args:
        parameters: dictionary of parameters of this training run

    Returns:
        the run folder

    """
    init_cuda_environment(parameters)
    return get_run_folder(parameters, create_folder=True)


def init_model_and_optimizer(run_folder: str, encoding: any, n_neurons: int, activation: any, model_type: str,
                             omega: float, hidden_layers: int, learning_rate: float):
    """Initializes the model and the associated training utility

    Args:
        run_folder: the folder where to store the model in case of early stopping
        encoding: encoding instance to use for the network
        n_neurons: the number of neurons per layer
        activation: activation function for the last network layer
        model_type: the model type
        omega: Omega value for siren activations
        hidden_layers: the number of hidden layer
        learning_rate: the utilized learning rate for the optimizer

    Returns:
        Tuple of model, the early_stopper, optimizer, scheduler

    """
    # Initializes the model
    model = network_initalizer.init_network(
        encoding,
        n_neurons=n_neurons,
        activation=activation,
        model_type=model_type,
        siren_omega=omega,
        hidden_layers=hidden_layers
    )

    # Initializes training utility: Early Stopping
    early_stopper = EarlyStopping(
        save_folder=run_folder
    )
    # Initializes training utility: Adam Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    # Initializes training utility: Scheduler for Learning Rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=200,
        min_lr=1e-6,
        verbose=True
    )

    return model, early_stopper, optimizer, scheduler


def init_training_sampler(sample: str, target_sample_method: str, sample_domain: [float], batch_size: int):
    """Creates a new target point sample with the given method and sample domain.

    Args:
        sample: the sample body's name
        target_sample_method: the sample method (e.g. 'spherical')
        sample_domain: the sample domain, specifies the sampling radius.
        batch_size: the number of points per function call

    Returns:
        sampling function

    """
    return get_target_point_sampler(
        batch_size,
        method=target_sample_method,
        bounds=sample_domain,
        limit_shape_to_asteroid=f"3dmeshes/{sample}_lp.pk"
    )
