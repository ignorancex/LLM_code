#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-10 19:08
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-18 18:26
Description        : Arguments for training.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import add_start_docstrings, AutoConfig, TrainingArguments, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from utils import logging_rank

logger = logging.getLogger('DeepSpeed')


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class CompactLLMTrainingArguments(TrainingArguments):
    """
    Arguments for CompactLLMTrainingArguments.
    """

    # Model arguments
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The model checkpoint / config for structure and weights initialization."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "model type",
        },
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": "attention type",
            "choices": ["flash_attention_2", "sdpa", "eager"]
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help":
            ("Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
             "dtype will be automatically derived from the model's weights."),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    from_config: bool = field(
        default=False,
        metadata={"help": "Load model from config, without weight"})
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={"help": ("trust remote code")},
    )

    # PEFT arguments
    peft_type: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "PEFT type, support [lora, p_tuning, prompt_tuning, prefix_tuning].",
            "choices": [
                "lora", "p_tuning", "prompt_tuning", "prefix_tuning", "None",
                "none"
            ]
        },
    )
    peft_config: Optional[str] = field(
        default=None,
        metadata={"help": "adapter config file."},
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "adapter path."},
    )

    # Dataset / Tokenization arguments
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."})
    data_type: Optional[str] = field(
        default="instruction_tuning",
        metadata={
            "help": "dataset type",
            "choices": ["pretraining", "instruction_tuning"]
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    padding: Optional[bool] = field(
        default=False,
        metadata={"help": "padding to max length in dataset"},
    )
    padding_side: Optional[str] = field(
        default="right",
        metadata={"help": "Padding side of tokenizer"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where to store intermediate results like the tokenizer cache."
        },
    )
    train_on_prompt: bool = field(
        default=True,
        metadata={"help": "Whether to train on previous round in dialogue."},
    )

    # Training arguments
    training: bool = field(default=True,
                           metadata={"help": "Train or inference"})
    profile: bool = field(
        default=False,
        metadata={"help": "do profile in the first few steps for debug"})
    min_learning_rate: float = field(default=1e-6,
                                     metadata={"help": "Min learning rate"})

    # Saving arguments
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "overwrite output dir"})
    save_last_interval: Optional[int] = field(
        default=3600,
        metadata={"help": ("interval to save latest checkpoint")},
    )

    # Supervised pinpoint tuning arguments
    path_patching_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Path to path patching results, used for pinpoint tuning. If None, no pinpoint tuning will be performed."
        },
    )
    precise_level: int = field(
        default=0,
        metadata={
            "help":
            "Precise level used in pinpoint tuning, higher means freeze more parameters. Here is a list of trainable parameters\n"
            "- 0: all, no pinpoint tuning.\n"
            "- 1: qkv_proj + o_proj + mlp + wte/lm_head.\n"
            "- 2: qkv_proj + o_proj + mlp.\n"
            "- 3: qkv_proj + o_proj.\n"
            "- 4: qkv_proj."
        },
    )
    train_topk: int = field(
        default=32,
        metadata={
            "help":
            "How many heads to train, used in pinpoint tuning. If None, no pinpoint tuning will be performed."
        },
    )
    train_kv: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to train key-value pairs, used in pinpoint tuning. If False, only train query projection matrix."
        },
    )


def parse_args(verbose: bool = True) -> CompactLLMTrainingArguments:
    """Parse arguments.

    Parameters
    ----------
    verbose : bool, optional
        Whether to output messages when parsing, by default True

    Returns
    -------
    CompactLLMTrainingArguments
        Parsed arguments.
    """
    
    parser = HfArgumentParser((CompactLLMTrainingArguments, ))
    (args, ) = parser.parse_args_into_dataclasses()

    # Set some arguments to None if they are set to "none" or "None"
    for arg in vars(args):
        value = getattr(args, arg)

        if value is not None and isinstance(value,
                                            str) and value.lower() == 'none':
            setattr(args, arg, None)

    if args.lr_scheduler_type == "polynomial":
        # If polynomial decay is used, we use a linear decay
        args.lr_scheduler_kwargs = {
            "power": 1.0,
            "lr_end": args.min_learning_rate
        }
    elif args.lr_scheduler_type == "cosine_with_min_lr":
        args.lr_scheduler_kwargs = {"min_lr": args.min_learning_rate}

    try:
        local_rank = int(os.environ['LOCAL_RANK'])
    except:
        local_rank = 0

    try:
        torch.cuda.set_device(local_rank)
    except:
        logging_rank("There is no CUDA device available.")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )

    # Set default cache dir
    if args.cache_dir is None:
        args.cache_dir = 'default_dataset_cache'

    # Detecting last checkpoint.
    last_checkpoint = None
    if args.resume_from_checkpoint is not None and os.path.isdir(
            args.resume_from_checkpoint):
        last_checkpoint = args.resume_from_checkpoint
        if verbose:
            logging_rank(f"Resuming training from {last_checkpoint}")
    elif os.path.exists(args.output_dir):

        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None or args.overwrite_output_dir > 0:
            last_checkpoint = None
            if verbose:
                logging_rank(
                    f"Output directory ({args.output_dir}) already exists, and will be overwrited "
                )
        elif last_checkpoint is not None:
            if verbose:
                logging_rank(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    args.resume_from_checkpoint = last_checkpoint

    # Set model type according to config
    config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code)
    args.model_type = config.model_type

    # Logging arguments
    args_str = "\n" + "#" * 40 + "Training Arguments" + "#" * 40 + "\n"
    for i, arg in enumerate(vars(args)):
        name = str(arg).replace("\n", "↵")
        value = str(getattr(args, arg)).replace("\n", "↵")
        args_str += f"[{i:03d}] {name}" + '.' * max(
            92 - len(name) - len(value), 5) + f"{value}\n"
    args_str += "#" * 98 + "\n"

    if verbose:
        logging_rank(args_str)

    return args
