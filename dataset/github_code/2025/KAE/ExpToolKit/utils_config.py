from typing import Any, Dict, Tuple
import yaml

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from . import models, utils_train
from .const import DATASET_CONST, MODEL_CONST, TRAIN_CONST

from DenseLayerPack import DENSE_LAYER_CONST


def load_config(config_path: str) -> dict:
    """
    Load config from yaml file.

    Args:
        config_path: Path to config file.

    Returns:
        dict: Config dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_model(config: dict) -> models.BaseAE:
    """
    Create model based on the config.

    Args:
        config: Config dictionary.

    Returns:
        models.BaseAE: Model instance.

    Raises:
        NotImplementedError: If dataset type or model type is not specified in config.
    """
    if config["DATA"]["type"] == DATASET_CONST.DATASET_MNIST:
        input_dim = DATASET_CONST.MNIST_INPUT_DIM
    elif config["DATA"]["type"] == DATASET_CONST.DATASET_FASHION_MNIST:
        input_dim = DATASET_CONST.FASHION_MNIST_INPUT_DIM
    elif config["DATA"]["type"] == DATASET_CONST.DATASET_CIFAR10:
        input_dim = DATASET_CONST.CIFAR10_INPUT_DIM
    elif config["DATA"]["type"] == DATASET_CONST.DATASET_CIFAR100:
        input_dim = DATASET_CONST.CIFAR100_INPUT_DIM
    else:
        raise NotImplementedError("dataset type is not specified in config")

    if "hidden_dims" in config["MODEL"]:
        hidden_dims = config["MODEL"]["hidden_dims"]
    else:
        hidden_dims = []

    if "latent_dim" in config["MODEL"]:
        latent_dim = config["MODEL"]["latent_dim"]
    else:
        raise NotImplementedError("latent_dim is not specified in config")

    kwargs = {}
    if "layer_type" in config["MODEL"]:
        layer_type = config["MODEL"]["layer_type"]
        if layer_type not in DENSE_LAYER_CONST.LAYER_TYPES:
            raise NotImplementedError(
                "layer_type is not defined in DENSE_LAYER_CONST.LAYER_TYPES"
            )
        elif layer_type == DENSE_LAYER_CONST.KAE_LAYER:
            if "order" in config["MODEL"]:
                order = config["MODEL"]["order"]
            else:
                order = DENSE_LAYER_CONST.TAYLOR_KAN_LAYER_DEFAULT_ORDER
            kwargs["order"] = order
    else:
        layer_type = DENSE_LAYER_CONST.LINEAR_LAYER

    if config["MODEL"]["model_type"] == MODEL_CONST.MODEL_AE:
        model = models.StandardAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            layer_type=layer_type,
            **kwargs
        )
    else:
        raise NotImplementedError("model type is not specified in config")

    return model


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """
    Create the optimizer based on the config.

    Args:
    - model (nn.Module): the model to be optimized.
    - config (dict): the config dictionary.

    Returns:
    - optimizer (optim.Optimizer): the optimizer object.

    If the config does not contain the "lr" and "weight_decay" keys,
    the default values are used.
    """
    if "lr" in config["TRAIN"]:
        lr = config["TRAIN"]["lr"]
    else:
        lr = 0.001

    if "weight_decay" in config["TRAIN"]:
        weight_decay = config["TRAIN"]["weight_decay"]
    else:
        weight_decay = 0.0005

    if "optim_type" not in config["TRAIN"]:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["TRAIN"]["optim_type"] == TRAIN_CONST.SGD_OPTIM:
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif config["TRAIN"]["optim_type"] == TRAIN_CONST.ADAM_OPTIM:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif config["TRAIN"]["optim_type"] == TRAIN_CONST.ADAMW_OPTIM:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError("optim type is not defined in TRAIN_CONST")

    return optimizer


def create_dataloader(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loader based on the config.

    Args:
        config (dict): configuration dictionary which must contain the "DATA" key.
            The "DATA" key must contain the "type" key and the "preprocessing" key is optional.
            The "type" key is the dataset type, and the "preprocessing" key is the preprocessing type.

    Returns:
        Tuple[DataLoader, DataLoader]: a tuple of two data loaders, the first one is the training data loader,
            and the second one is the test data loader.
    """
    if "batch_size" in config["TRAIN"]:
        batch_size = config["TRAIN"]["batch_size"]
    else:
        batch_size = 128

    if "type" in config["DATA"]:
        dataset_name = config["DATA"]["type"]
        if dataset_name not in DATASET_CONST.DATASET_TYPES:
            raise NotImplementedError(
                "dataset type is not defined in DATASET_CONST.DATASET_TYPES"
            )
    else:
        raise NotImplementedError("dataset type is not specified in config")

    if "preprocessing" in config["DATA"]:
        preprocessing = config["DATA"]["preprocessing"]
    else:
        preprocessing = DATASET_CONST.PREPROCESSING_ORIGINAL
    if preprocessing not in DATASET_CONST.PREPROCESSING_METRIC:
        raise NotImplementedError(
            "preprocessing is not defined in DATASET_CONST.PREPROCESSING_METRIC"
        )

    train_loader, test_loader = utils_train.dataset_loader(
        dataset_name=dataset_name, batch_size=batch_size, preprocessing=preprocessing
    )
    return train_loader, test_loader


def create_dataloader_by_parameter(
    data_type: str,
    batch_size: int,
    preprocessing: str = DATASET_CONST.PREPROCESSING_ORIGINAL,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loader based on the input parameters.

    Args:
        data_type (str): dataset type.
        batch_size (int): batch size.
        preprocessing (str, optional): preprocessing type. Defaults to DATASET_CONST.PREPROCESSING_ORIGINAL.

    Returns:
        Tuple[DataLoader, DataLoader]: a tuple of two data loaders, the first one is the training data loader,
            and the second one is the test data loader.
    """
    if data_type not in DATASET_CONST.DATASET_TYPES:
        raise NotImplementedError(
            "dataset type is not defined in DATASET_CONST.DATASET_TYPES"
        )
    train_loader, test_loader = utils_train.dataset_loader(
        dataset_name=data_type, batch_size=batch_size, preprocessing=preprocessing
    )
    return train_loader, test_loader


def create_train_setting(
    config: dict,
) -> Dict[str, Any]:
    """
    Create a dictionary containing the required parameters for training a model.

    Args:
    - config (dict): configuration dictionary which must contain the "DATA" key.
        The "DATA" key must contain the "type" key and the "preprocessing" key is optional.
        The "type" key is the dataset type, and the "preprocessing" key is the preprocessing type.

    Returns:
    - kwargs (dict): a dictionary containing the required parameters for training a model.
    """
    if "random_seed" in config["TRAIN"]:
        random_seed = config["TRAIN"]["random_seed"]
    else:
        random_seed = 2024

    if "epochs" in config["TRAIN"]:
        epochs = config["TRAIN"]["epochs"]
    else:
        epochs = 10

    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_loader, test_loader = create_dataloader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {
        "model": model,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "optimizer": optimizer,
        "epochs": epochs,
        "device": device,
        "random_seed": random_seed,
    }

    return kwargs
