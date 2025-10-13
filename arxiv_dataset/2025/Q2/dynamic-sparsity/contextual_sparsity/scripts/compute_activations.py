# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from contextual_sparsity.utils.layer_names import FC_DOWN, FC_UP, get_layer_ids
from contextual_sparsity.utils.misc import parse_dtype
from contextual_sparsity.utils.stats import StopComputation, collect_activations

log = logging.getLogger(__name__)
DATAFILE = "activations.h5"
MEMORY_OPTIONS = ["cpu", "cuda", "disk"]


def collect_input_and_output(
    module,
    args,
    kwargs,
    out,
    layer_name,
    data,
    store_input,
    store_output,
    dtype,
    stop_computation=False,
):
    """
    Hook function used to store input/outputs of each layer
    """
    batch_size = out.shape[0]

    # Collect the inputs of the layer
    if store_input:
        i = args[0].cpu()
        i = i.type(dtype)
        i = i.view(batch_size, -1, i.shape[-1])
        data[f"{layer_name}.input"] = i

    # Collect the outputs of the layer
    if store_output:
        o = out.cpu()
        o = o.type(dtype)
        o = o.view(batch_size, -1, o.shape[-1])
        data[f"{layer_name}.output"] = o

    if stop_computation:
        # skip all computation once the last value is saved
        raise StopComputation()


def output_to_next_input(
    out: Any, in_args: List[Any], in_kwargs: Dict[str, Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    return [out[0]], in_kwargs


def compute_activations(
    dataloader: DataLoader,
    dtype: Union[str, torch.dtype],
    preprocess_batch: Optional[Callable],
    dense_model: torch.nn.Module,
    activation_ids: List[str],
    memory: str,
    stop_computation: bool = True,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    compute all the specified activations for a given model and dataloader. Activations are cast to the specified dtype
    and stored to disk, gpu, or cpu memory.
    """
    assert memory in MEMORY_OPTIONS

    batch_size = dataloader.batch_size
    # Parse the data type
    dtype = parse_dtype(dtype)
    layers = {}
    layer_id = ""

    # Consider all the activations ids and determine for each layer if we collect only inputs, outputs or both
    for activation_id in activation_ids:
        layer_id = ".".join(activation_id.split(".")[:-1])
        io = activation_id.split(".")[-1]
        assert dense_model.get_submodule(layer_id) is not None, f"{layer_id} not found in net."
        assert io in [
            "input",
            "output",
        ], f"{activation_id} is not a valid activation id."
        if layer_id not in layers:
            layers[layer_id] = {"store_input": False, "store_output": False}
        layers[layer_id][f"store_{io}"] = True
        layers[layer_id]["dtype"] = dtype

    if len(layers) > 0:
        layers[layer_id]["stop_computation"] = stop_computation

    # Consider an appropriate hook based on if we are interested in storing only input/output or both
    collection_funcs = {}
    for activation_id in activation_ids:
        layer_id = ".".join(activation_id.split(".")[:-1])
        collection_funcs[layer_id] = partial(collect_input_and_output, **layers[layer_id])

    dense_model.eval()

    if memory == "disk":
        log.info(f"Storing the activations in {os.getcwd()}")
        activations = h5py.File(DATAFILE, "w")
    else:
        activations = {}

    log.info("Computing the activations")
    # Store all the activations (in memory or h5py file)
    last_idx = 0
    dataset_size = len(dataloader.dataset)
    with torch.no_grad():
        with collect_activations(
            collection_funcs=collection_funcs,
            dense_model=dense_model,
            preprocess_batch=preprocess_batch,
        ) as collect_act:
            for batch in tqdm(dataloader):
                acts = collect_act(batch)
                for act_name, act_value in acts.items():
                    # For model that use shape [seq_len, batch_size, features], we transpose the first two dimensions
                    if act_value.shape[0] != dataloader.batch_size:
                        act_value = act_value.transpose(1, 0)

                    assert act_value.shape[0] == dataloader.batch_size

                    if act_name not in activations:
                        # Determine the shape of the whole dataset and allocate it
                        shape = [dataset_size] + list(act_value.shape[1:])
                        if memory == "disk":
                            activations.create_dataset(
                                act_name,
                                shape=shape,
                                dtype=act_value.data.numpy().dtype,
                            )
                        else:
                            activations[act_name] = torch.zeros(
                                shape, dtype=act_value.dtype, device=memory
                            )
                    # Set the values for each entry
                    for i in range(act_value.shape[0]):
                        activations[act_name][last_idx + i] = act_value[i].detach()
                last_idx += batch_size

    # Save the data to file if required
    if memory == "disk":
        activations.flush()
        log.info(f"Storing the activations in {os.path.join(os.getcwd(), DATAFILE)}")

    return activations


def store_activations_main(
    conf: DictConfig,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Function to store the activations to disk given the specified configuration. This function is called when
    specifying experiment=store_activations from CLI
    """

    log.info("Instantiating the original Dataset")
    split = conf.activations.split
    dataloader = instantiate(conf.data[split], tokenizer=conf.tokenizer, shuffle=False)
    preprocess_batch = instantiate(conf.preprocess_batch)

    log.info("Instantiating the Model")
    dense_model = instantiate(conf.dense_model).to(conf.device)

    activation_ids = conf.activations.activation_ids

    # If "none" store input and output of down layers and input of up layers
    if activation_ids is None:
        activations_ids = get_layer_ids(
            model_id=conf.model_id,
            layer_type=FC_DOWN,
            layer_names=conf.activations.layer_ids,
        ) + get_layer_ids(
            model_id=conf.model_id,
            layer_type=FC_UP,
            layer_names=conf.activations.layer_ids,
        )
    else:
        activations_ids = activation_ids

    if isinstance(activations_ids, str):
        activations_ids = [activations_ids]

    compute_activations(
        dataloader=dataloader,
        dtype=conf.activations.dtype,
        preprocess_batch=preprocess_batch,
        dense_model=dense_model,
        activation_ids=activations_ids,
        memory="disk",
    )
