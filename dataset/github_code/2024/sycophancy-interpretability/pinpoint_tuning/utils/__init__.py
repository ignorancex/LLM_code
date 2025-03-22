#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-11 11:24
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:26
Description        : __init__.py
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import json
import logging
import os
import shutil
from typing import List

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.utils import is_peft_available
from transformers.utils.logging import set_verbosity_info, enable_default_handler, enable_explicit_format

IGNORE_INDEX = -100

SUPPORTED_MODEL_CLASSES = (PreTrainedModel, ) if not is_peft_available() else (
    PreTrainedModel, PeftModel)

logger = logging.getLogger('DeepSpeed')


def transformers_logging_setup() -> None:
    """Setup logging basics for transformers.
    """
    
    set_verbosity_info()
    enable_default_handler()
    enable_explicit_format()

    return


def logging_rank(message: str, rank: int = 0) -> None:
    """Log message on the rank provided.

    Parameters
    ----------
    message : str
        Message to log.
    rank : int, optional
        Rank used to log, by default 0
    """
    
    if not dist.is_initialized():
        logger.info(message)
        return

    if dist.get_rank() == rank:
        logger.info(message)

    return


def delete_zero_hf(output_dir: str) -> None:
    """Delete zero states in previous output checkpoints.

    Parameters
    ----------
    output_dir : str
        Output directory whose zero states will be deleted.
    """
    
    if torch.distributed.get_rank() == 0:
        if os.path.exists(output_dir):
            for file_name in os.listdir(output_dir):
                if "global_step" in file_name:
                    shutil.rmtree(os.path.join(output_dir, file_name))

    return


def read_jsonl(path: str) -> List[dict]:
    """Read jsonl file.

    Parameters
    ----------
    path : str
        Path to jsonl file.

    Returns
    -------
    List[dict]
        The list of datapoints read from jsonl file.
    """

    return [json.loads(line) for line in open(path, 'r', encoding='utf-8')]


def write_jsonl(datapoints: List[dict],
                path: str,
                overwrite: bool = True) -> None:
    """Write datapoints to jsonl file.

    Parameters
    ----------
    datapoints : List[dict]
        Datapoints to be saved.
    path : str
        Path to jsonl file.
    overwrite : bool, optional
        Whether to overwrite existing file, by default True
    """
    
    base_dir = os.path.dirname(path)

    # Create directory if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(path, 'w' if overwrite else 'a', encoding='utf-8') as file:
        for datapoint in datapoints:
            file.write(json.dumps(datapoint, ensure_ascii=False) + "\n")
