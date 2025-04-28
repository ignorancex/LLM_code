import os
import pickle
from typing import List, Tuple, Dict

import torch
import torch.nn as nn

from gravann.network import network_initalizer, encodings, layers
from gravann.util import deep_get


def read_models(root_directory: os.PathLike) -> List[Tuple[nn.Module, Dict]]:
    """Reads all models and their configuration from a given root directory. Each "best_model.mdl" file needs
    to be in the same directory as the "config.pk in order to be associated with it.

    Args:
        root_directory: the directory from where to recursively start searching

    Returns:
        list of tuples of neural network and configuration

    """
    models = []
    for dir_path, _, filenames in os.walk(root_directory):
        model_stats, config = None, None
        for filename in filenames:
            path = os.path.join(dir_path, filename)
            if filename.endswith("best_model.mdl"):
                model_stats = torch.load(path)
            elif filename.endswith("config.pk"):
                config = _read_config_file(path)
        if model_stats is not None and config is not None:
            populated_model = _populate_model(model_stats, config)
            models.append((populated_model, config))
    return models


def _read_config_file(path: str) -> dict:
    """Reads a pickled config from file

    Args:
        path: the path to the file

    Returns:
        dictionary of the config

    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def _populate_model(model_stats: dict, config: dict) -> nn.Module:
    """Initializes a model from the given neuron weights in a dict model_states and a given config.

    Args:
        model_stats: the neurons' weights
        config: the configuration of the model

    Returns:
        the populated model

    """
    encoding = encodings.get_encoding(deep_get(config, ["Encoding", "encoding"]))
    activation = layers.get_activation_layer(deep_get(config, ["Activation", "activation"]))
    model = network_initalizer.init_network(
        encoding,
        model_type=deep_get(config, ["model_type", "Model"]),
        activation=activation,
        n_neurons=config["n_neurons"],
        hidden_layers=config["hidden_layers"],
        siren_omega=config.get("omega", 30.0)  # Omega was not always saved, but it was always 30.0
    )
    model.load_state_dict(model_stats)
    return model
