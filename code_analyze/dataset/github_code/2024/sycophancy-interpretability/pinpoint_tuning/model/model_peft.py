#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:12
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:18
Description        : PEFT model builder.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import json
import os
from argparse import Namespace

import torch
from peft import PeftModelForCausalLM
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, _prepare_prompt_learning_config
from peft.tuners import (
    LoraConfig,
    AdaLoraConfig,
    IA3Config,
    AdaptionPromptConfig,
    PromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from transformers import PreTrainedModel

from utils import logging_rank

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "adalora": AdaLoraConfig,
    "ia3": IA3Config,
    "adaption_prompt": AdaptionPromptConfig,
    "prompt_tuning": PromptTuningConfig,
    "prefix_tuning": PrefixTuningConfig,
    "p_tuning": PromptEncoderConfig,
}


def build_peft_from_model(args: Namespace,
                          model: PreTrainedModel) -> PeftModelForCausalLM:
    """Build peft model from pretrained model path.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    model : PreTrainedModel
        Base model.

    Returns
    -------
    PeftModelForCausalLM
        Peft model.
    """

    # Only support inference mode
    assert args.peft_path and os.path.exists(
        args.peft_path), "Peft path must be specified."
    assert args.peft_type in PEFT_TYPE_TO_CONFIG_MAPPING, f"{args.peft_type} is not supported currently, only support these peft methods: [{PEFT_TYPE_TO_CONFIG_MAPPING.keys()}]"

    torch_dtype = (args.torch_dtype if args.torch_dtype in ["auto", None] else
                   getattr(torch, args.torch_dtype))
    logging_rank(
        f"Creating peft model of {args.peft_type} from path {args.peft_path}")

    max_memory = {
        i: torch.cuda.mem_get_info(i)[1]
        for i in range(torch.cuda.device_count())
    }
    model = PeftModelForCausalLM.from_pretrained(model,
                                                 args.peft_path,
                                                 torch_dtype=torch_dtype,
                                                 max_memory=max_memory,
                                                 device_map="auto")

    return model


def build_peft_from_config(args: Namespace,
                           model: PreTrainedModel) -> PeftModelForCausalLM:
    """Build peft model from peft config.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    model : PreTrainedModel
        Base model.

    Returns
    -------
    PeftModelForCausalLM
        Peft model.
    """

    # Support training from scratch but only single PEFT model
    assert args.peft_config and os.path.exists(
        args.peft_config), "Peft config must be specified."
    assert args.peft_type in PEFT_TYPE_TO_CONFIG_MAPPING, f"{args.peft_type} is not supported currently, only support these peft methods: [{PEFT_TYPE_TO_CONFIG_MAPPING.keys()}]"

    logging_rank(
        f"Creating peft model of {args.peft_type} from config {args.peft_config}"
    )
    peft_config_dict = json.load(open(args.peft_config))

    peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[args.peft_type](
        task_type="CAUSAL_LM")
    for key, value in peft_config_dict.items():
        if hasattr(peft_config, key):
            logging_rank(f"Setting {key} to {value}")
            setattr(peft_config, key, value)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)

    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get(
        "name_or_path", None)

    if hasattr(peft_config,
               "is_prompt_learning") and peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config,
                                                      model_config)

    model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        model, peft_config)
    # model.print_trainable_parameters()

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logging_rank(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

    return model


def build_model_peft(args: Namespace,
                     model: PreTrainedModel) -> PeftModelForCausalLM:
    """Build peft model from pretrained model path or config.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    model : PreTrainedModel
        Base model.

    Returns
    -------
    PeftModelForCausalLM
        Peft model.

    Raises
    ------
    ValueError
        Pert type not implemented.
    """

    # If peft is not implemented, return original model
    if args.peft_type not in PEFT_TYPE_TO_CONFIG_MAPPING:
        logging_rank(
            f"Peft type {args.peft_type} not implemented, returning original model"
        )
        return model

    if hasattr(args, "peft_path") and args.peft_path and os.path.exists(
            args.peft_path):
        return build_peft_from_model(args, model)
    elif hasattr(args, "peft_config") and args.peft_config and os.path.exists(
            args.peft_config):
        return build_peft_from_config(args, model)
    else:
        raise ValueError("Either peft_path or peft_config must be specified.")
