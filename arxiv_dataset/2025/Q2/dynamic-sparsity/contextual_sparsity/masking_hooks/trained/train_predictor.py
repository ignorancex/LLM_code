# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from contextual_sparsity.data.activations import ActivationDictDataset
from contextual_sparsity.evaluation.predictor import evaluate_predictor
from contextual_sparsity.masking_hooks.trained.model import Predictor
from contextual_sparsity.masking_hooks.trained.optimization import optimize
from contextual_sparsity.scripts.compute_activations import compute_activations
from contextual_sparsity.utils.layer_names import (
    FC_ACT,
    FC_DOWN,
    FC_GATE,
    FC_UP,
    MODEL_MAPS,
    get_block_id,
    get_layer_id,
    has_gate,
)
from contextual_sparsity.utils.misc import split_gate_up_layer

log = logging.getLogger(__name__)

PREDICTOR_FILENAME = "predictor.pyt"
PREDICTOR_CONF_FILENAME = "predictor_conf.yaml"


def delete_private_keys(config: Any) -> Any:
    """
    Remove all keys starting with __ from a spefified configuration
    """
    if not isinstance(config, DictConfig):
        return config
    else:
        updated_config = {}
        for key, val in config.items():
            if not key.startswith("__"):
                updated_config[key] = delete_private_keys(val)
        return OmegaConf.create(updated_config)


def save_predictor(
    save_path: str,
    predictor: Predictor,
    predictor_conf: DictConfig,
    logs: Dict[str, pd.DataFrame],
):
    """
    Save a predictor (and train logs) to a specified file.
    """
    # Save the predictor config
    conf_filepath = os.path.abspath(os.path.join(save_path, PREDICTOR_CONF_FILENAME))

    # Delete the keys that start with __
    predictor_conf = delete_private_keys(predictor_conf)

    OmegaConf.save(
        predictor_conf,
        conf_filepath,
    )
    log.info(f"Predictor configuration stored at {conf_filepath}")

    # Save the predictor
    state_dict = {"state_dict": predictor.state_dict()}
    state_dict.update(
        {
            "input_dim": predictor.input_dim,
            "output_dim": predictor.output_dim,
        }
    )
    model_filepath = os.path.join(save_path, PREDICTOR_FILENAME)
    torch.save(state_dict, model_filepath)
    log.info(f"Predictor stored at {model_filepath}")

    for name, df in logs.items():
        log_filepath = os.path.abspath(os.path.join(save_path, f"{name}.csv"))
        log.info(f"{name} log stored at {log_filepath}")
        df.to_csv(log_filepath)


def load_predictor(
    model_conf: DictConfig,
    filepath: str,
) -> Predictor:
    """
    Load a predictor from a specified file.
    """
    predictor_dict = torch.load(filepath)
    predictor = instantiate(
        model_conf,
        input_dim=predictor_dict["input_dim"],
        output_dim=predictor_dict["output_dim"],
    )
    predictor.load_state_dict(predictor_dict["state_dict"])
    return predictor


def resolve_layer_ids(
    model_id: str,
    layer_to_mask: Union[int, str],
    input_activation: Union[int, str],
) -> Tuple[str, str, str, str]:
    """
    Determine the layer_id, activation id of the input and output of the predictor given the list of layers to mask.
    """
    up_name = MODEL_MAPS[model_id][FC_UP]
    down_name = MODEL_MAPS[model_id][FC_DOWN]

    if isinstance(layer_to_mask, int):
        sparse_layer_id = get_block_id(model_id=model_id, block_name=layer_to_mask)
    else:
        assert isinstance(layer_to_mask, str)
        sparse_layer_id = layer_to_mask

    if isinstance(input_activation, int):
        input_activation_id = (
            get_layer_id(model_id=model_id, layer_type=FC_UP, layer_name=input_activation)
            + ".input"
        )
    elif not input_activation.endswith(".input") and not input_activation.endswith(".output"):
        if not input_activation.endswith(up_name) and not input_activation.endswith(down_name):
            # We have the name of a block, so, by default we take the input of FC_UP
            input_activation_id = input_activation + up_name + ".input"
        else:
            # We have the name of a fc_up or fc_down layer, so we consider its input
            input_activation_id = input_activation + ".input"
    else:
        # We have the full reference to a layer
        assert input_activation.endswith(".input") or input_activation.endswith(".output")
        input_activation_id = input_activation

    sparse_activation_id = ".".join([sparse_layer_id, down_name, "input"])

    last_input_id = ".".join([sparse_layer_id, up_name, "input"])

    return sparse_layer_id, input_activation_id, sparse_activation_id, last_input_id


def get_activation_dataloader(
    data_conf: DictConfig,
    dense_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    activation_map: Dict[str, str],
) -> DataLoader:
    """ "
    Make an activation dataloader from a specified model, tokenizer and data configuration
    """

    original_dataloader = instantiate(data_conf.activation_data.dataloader, tokenizer=tokenizer)

    # Compute all the activations
    activations = compute_activations(
        dataloader=original_dataloader,
        dense_model=dense_model,
        dtype=data_conf.activation_data.dtype,
        memory=data_conf.activation_data.memory,
        activation_ids=list(sorted(activation_map.values())),
        preprocess_batch=instantiate(data_conf.activation_data.preprocess_batch),
    )

    # Make a dataset
    dataset = ActivationDictDataset(
        activations=activations,
        flatten=data_conf.activation_data.flatten_activations,
        **activation_map,
    )

    # And then a corresponding dataloader
    num_workers = data_conf.num_workers if "num_workers" in data_conf is not None else 0
    shuffle = data_conf.shuffle if "shuffle" in data_conf is not None else False
    activation_dataloader = DataLoader(
        dataset=dataset,
        batch_size=data_conf.batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return activation_dataloader


def train_and_store_predictor(
    predictor_conf: DictConfig,
    model_id: str,
    dense_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    save_path: str,
) -> float:
    """
    Instantiate, train and then store a predictor. predictor_conf contains all the predictor, training
    and data parameters. Training using the same predictor_conf will result in the same trained predictors.
    """

    #########################
    # Layer Name Resolution #
    #########################
    # Resolve the ids for layer to mask and input_id to be strings
    sparse_layer_id, input_activation_id, sparse_activation_id, last_input_id = resolve_layer_ids(
        model_id=model_id,
        layer_to_mask=predictor_conf.layer_to_mask,
        input_activation=predictor_conf.input_activation,
    )

    activation_map = {"x": input_activation_id}

    # If input and output are not from the same layer, store the predictor input and the input to the up and gate
    # layers for the layer that we are sparsifying

    if input_activation_id != last_input_id:
        activation_map["x_sparse"] = last_input_id

    ############################
    # Dataloader Instantiation #
    ############################
    train_loader = get_activation_dataloader(
        data_conf=predictor_conf.data.train,
        dense_model=dense_model,
        tokenizer=tokenizer,
        activation_map=activation_map,
    )

    valid_loader = get_activation_dataloader(
        data_conf=predictor_conf.data.valid,
        dense_model=dense_model,
        tokenizer=tokenizer,
        activation_map=activation_map,
    )

    ####################
    # Layer Extraction #
    ####################
    # Get the up, down, activation function (and gate) from the model
    layer_components = {
        "up_layer": MODEL_MAPS[model_id][FC_UP],
        "down_layer": MODEL_MAPS[model_id][FC_DOWN],
        "act_fn": MODEL_MAPS[model_id][FC_ACT],
    }
    if has_gate(model_id):
        layer_components["gate_layer"] = MODEL_MAPS[model_id][FC_GATE]

    # Determine a reference to the down layer following the considered activations and retrieve it from the net
    layer_ids = {
        name: ".".join([sparse_layer_id, layer_type])
        for name, layer_type in layer_components.items()
    }

    modules = {name: dense_model.get_submodule(layer_id) for name, layer_id in layer_ids.items()}
    # The Up and Gate are in one single layer, so we need to split it
    if has_gate(model_id) and MODEL_MAPS[model_id][FC_UP] == MODEL_MAPS[model_id][FC_GATE]:
        gate_up_layer = modules["up_layer"]
        assert isinstance(gate_up_layer, nn.Linear)
        modules["gate_layer"], modules["up_layer"] = split_gate_up_layer(gate_up_layer)

    # Determine the shape of input/output pairs
    down_layer = dense_model.get_submodule(
        get_layer_id(model_id=model_id, layer_type=FC_DOWN, layer_name=0)
    )

    assert isinstance(down_layer, nn.Linear)
    input_dim = down_layer.weight.shape[0]
    output_dim = down_layer.weight.shape[1]

    ###########################
    # Predictor Instantiation #
    ###########################
    log.info("Instantiating the predictor")
    predictor = instantiate(predictor_conf.model, input_dim=input_dim, output_dim=output_dim)

    loss = instantiate(predictor_conf.loss, predictor=predictor, **modules)
    log.info(loss)

    ######################
    # Predictor Training #
    ######################
    log.info("Training the predictor")
    train_log = optimize(
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_func=loss,
        **instantiate(predictor_conf.optimization),
    )

    ########################
    # Predictor Evaluation #
    ########################
    log.info("Evaluating the predictor")
    results = evaluate_predictor(
        predictor=predictor, dataloader=valid_loader, k_spacing=64, **modules
    )

    ##########
    # Saving #
    ##########
    predictor = loss.predictor
    save_predictor(
        save_path=save_path,
        predictor=predictor,
        predictor_conf=predictor_conf,
        logs={
            "train": train_log,
            "results": results,
        },
    )

    return 0.0


def get_candidate_predictor_dirs(
    predictor_conf: DictConfig,
    predictor_save_dir: str,
) -> List[str]:
    """
    Determine which path contain a cached predictor that is compatible with the given predictor_conf.
    """

    # Check all predictors in the specified folder
    valid_models = {}
    if not os.path.isdir(predictor_save_dir):
        os.makedirs(predictor_save_dir)
        return []

    for experiment in os.listdir(predictor_save_dir):
        base_dir = os.path.join(predictor_save_dir, experiment)
        config_filename = os.path.join(base_dir, PREDICTOR_CONF_FILENAME)
        predictor_filename = os.path.join(base_dir, PREDICTOR_FILENAME)

        if os.path.isfile(config_filename) and os.path.isfile(predictor_filename):
            valid_models[experiment] = {
                "config": config_filename,
                "base": base_dir,
            }

    valid_predictor_dirs = []
    for experiment, filepaths in valid_models.items():
        with open(filepaths["config"], "r") as f:
            saved_predictor_conf = OmegaConf.load(f)

        if predictor_conf == saved_predictor_conf:
            log.info(f"The configuration from the saved predictor {experiment} is equivalent.")
            valid_predictor_dirs.append(filepaths["base"])

    if len(valid_predictor_dirs) == 0:
        return []
    else:
        log.info(f"Found possible {len(valid_predictor_dirs)} valid predictors.")
        return valid_predictor_dirs


def get_predictor(
    predictor_conf: DictConfig,
    dense_model: nn.Module,
    model_id: str,
    tokenizer: PreTrainedTokenizer,
    predictor_cache_dir: str,
    force_retrain: bool = False,
) -> Predictor:
    """
    Obtain a predictor instance from a specified predictor_conf. This will be first searched into the cache
    (predictor_cache_dir). If there are no hits, a new predictor is trained and stored to disk.
    """

    predictor = None
    if not force_retrain:
        possible_predictor_dirs = get_candidate_predictor_dirs(
            predictor_conf=predictor_conf,
            predictor_save_dir=predictor_cache_dir,
        )

        # If any match is found, load the model. We use the last available occurrence
        for possible_predictor_dir in reversed(possible_predictor_dirs):
            try:
                predictor = load_predictor(
                    model_conf=predictor_conf.model,
                    filepath=os.path.join(possible_predictor_dir, PREDICTOR_FILENAME),
                )
                log.info(f"Predictor loaded from {possible_predictor_dir}")
                break
            except Exception as e:
                log.warning(str(e))

    if predictor is None:
        log.info("Predictor for the specified configuration not found.")

        # Make a new experiment id based on the date and time and the order
        current_date = datetime.now()
        experiment_id = current_date.strftime("%Y-%m-%d_%H-%M-%S.%f")
        save_path = os.path.join(predictor_cache_dir, experiment_id)

        # Create the experiment directory
        os.makedirs(save_path, exist_ok=False)

        # Train the predictors
        train_and_store_predictor(
            predictor_conf=predictor_conf,
            dense_model=dense_model,
            model_id=model_id,
            tokenizer=tokenizer,
            save_path=save_path,
        )

        # Load the saved model
        predictor = load_predictor(
            model_conf=predictor_conf.model,
            filepath=os.path.join(save_path, PREDICTOR_FILENAME),
        )

    return predictor
