#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:12
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:17
Description        : Huggingface model builder.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

from argparse import Namespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available

from utils import logging_rank


def build_model_hf(args: Namespace) -> AutoModelForCausalLM:
    """Build huggingface model.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.

    Returns
    -------
    AutoModelForCausalLM
        Huggingface model.
    """

    # Setup model parameters
    torch_dtype = (args.torch_dtype if args.torch_dtype in ["auto", None] else
                   getattr(torch, args.torch_dtype))
    device_map = None if hasattr(args,
                                 "training") and args.training else "auto"
    trust_remote_code = getattr(args, "trust_remote_code", True)

    # Implementation of attention, if not available, use fallback to eager attention
    attn_implementation = args.attn_implementation
    if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available(
    ):
        attn_implementation = "sdpa"
    if attn_implementation == "sdpa" and not is_torch_sdpa_available():
        attn_implementation = "eager"

    model_args = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
    }

    # Logging model arguments that are actually used
    logging_rank(model_args.__str__())

    # Load model from huggingface
    if hasattr(args, "from_config") and args.from_config:
        # Load model from config
        model = AutoModelForCausalLM.from_config(
            args.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code)
        model.config.attn_implementation = attn_implementation
    else:
        # Load model from pretrained weights
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )

    return model


def build_tokenizer_hf(args: Namespace) -> AutoTokenizer:
    """Build huggingface tokenizer.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.

    Returns
    -------
    AutoTokenizer
        Tokenizer.
    """

    # Load tokenizer from huggingface
    trust_remote_code = getattr(args, "trust_remote_code", True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=trust_remote_code,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    padding_side = getattr(args, "padding_side", "left")
    tokenizer.padding_side = padding_side

    return tokenizer
