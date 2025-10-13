# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import importlib
import inspect
import logging
from typing import Dict, List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from contextual_sparsity.utils.sparsify import build_sparse_model

log = logging.getLogger(__name__)


def get_parameters(class_path: str) -> List[str]:
    """
    Determine the parameters of the given function
    """
    module_name = ".".join(class_path.split(".")[:-1])
    function_name = class_path.split(".")[-1]
    func_builder = getattr(importlib.import_module(module_name), function_name)
    func_params = inspect.signature(func_builder).parameters
    return list(func_params)


def evaluate_llm_main(conf: DictConfig) -> Dict[str, float]:
    """
    Evaluate the LLM model according to the given configuration
    """
    torch.set_grad_enabled(False)

    log.info("Instantiating the tokenizer")
    tokenizer = instantiate(conf.tokenizer)

    log.info("Instantiating the dense model")
    dense_model = instantiate(conf.dense_model)

    # Determine if dense_model, calibration_data, and tokenizer are required to build the masking hooks
    # If so, pass them
    masking_hooks_parameters = get_parameters(conf.masking_hooks._target_)
    kwargs = {}

    # Instantiate the calibration
    calibration_data = instantiate(conf.data.calibration, tokenizer=tokenizer)

    if "calibration_data" in masking_hooks_parameters:
        kwargs["calibration_data"] = calibration_data
    if "dense_model" in masking_hooks_parameters:
        kwargs["dense_model"] = dense_model
    if "tokenizer" in masking_hooks_parameters:
        kwargs["tokenizer"] = tokenizer

    log.info("Instantiating the masking functions")
    masking_hooks = instantiate(conf.masking_hooks, **kwargs)

    # Create an instance of the hardware simulator, add cache hooks to masking hooks.
    hardware = None
    if conf.hw_simulator is not None:
        assert conf.cache_hooks is not None
        log.info("Instantiating Hardware Simulator")
        hardware = instantiate(conf.hw_simulator, model=dense_model, masking_hooks=masking_hooks)
        masking_hooks = instantiate(
            conf.cache_hooks, masking_hooks=masking_hooks, hw_simulator=hardware
        )

    log.info("Building the sparse model")
    sparse_model = build_sparse_model(
        masking_hooks=masking_hooks,
        dense_model=dense_model,
    )

    # Add the LoRA adapter if specified
    if "adapter" in conf:
        tokenizer = tokenizer
        from contextual_sparsity.adapters import add_adapters

        add_adapters(
            sparse_model=sparse_model,
            adapter_conf=conf.adapter,
        )

        if "load" in conf.adapter:
            from contextual_sparsity.adapters import load_adapters

            sparse_model = load_adapters(
                sparse_model=sparse_model,
                adapter_conf=conf.adapter,
                adapter_path=conf.adapter.load,
            )
        elif "training" in conf.adapter:
            from contextual_sparsity.adapters import train_adapters

            train_adapters(
                sparse_model=sparse_model,
                tokenizer=tokenizer,
                adapter_conf=conf.adapter,
            )

    log.info(sparse_model)

    # Perform all the specified evaluations
    merged_results = {}
    for name, evaluation in conf.evaluation.items():
        # If the tokenizer is required, pass it to the evaluation function
        kwargs = {}

        eval_parameters = get_parameters(evaluation._target_)
        if "tokenizer" in eval_parameters:
            kwargs["tokenizer"] = tokenizer
        if "test_data" in eval_parameters:
            # Replace the configuration with its instance
            kwargs["test_data"] = instantiate(evaluation.test_data, tokenizer=tokenizer)
            with open_dict(evaluation):
                del evaluation["test_data"]

        log.info(f"Running the {name} evaluation")
        results = instantiate(evaluation, model=sparse_model, hw_simulator=hardware, **kwargs)
        merged_results.update(results)

    # Print all the return values
    for quantity, value in merged_results.items():
        log.info(f"{quantity}: {value}")

    return merged_results
