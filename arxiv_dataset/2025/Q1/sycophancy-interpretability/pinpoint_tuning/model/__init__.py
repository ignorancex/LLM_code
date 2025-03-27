#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-10 19:09
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:17
Description        : __init__.py
-------- 
Copyright (c) 2024 Wei Chen. 
'''

from argparse import Namespace

from transformers import PreTrainedModel, PreTrainedTokenizer

from model.model_hf import build_model_hf, build_tokenizer_hf
from model.model_peft import build_model_peft


def build_model(args: Namespace) -> PreTrainedModel:
    """Build model from args

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.

    Returns
    -------
    PreTrainedModel
        Transformer model.
    """
    model = build_model_hf(args)

    # Build PEFT models
    if args.peft_type is not None:
        model = build_model_peft(args, model)

    # Enable gradient checkpointing
    if hasattr(args, "gradient_checkpointing") and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def build_tokenizer(args: Namespace) -> PreTrainedTokenizer:
    """Build tokenizer from args

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.

    Returns
    -------
    PreTrainedTokenizer
        Tokenizer.
    """
    tokenizer = build_tokenizer_hf(args)
    return tokenizer
