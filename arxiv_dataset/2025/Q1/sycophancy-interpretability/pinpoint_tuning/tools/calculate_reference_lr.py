#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:55
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:10
Description        : Brought and modified from LLaMA-Factory.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import sys

sys.path.append('..')

import os
import math

import deepspeed
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from dataset import build_dataset
from model import build_tokenizer
from utils import IGNORE_INDEX
from utils.arguments import parse_args
"""
EXAMPLE: Set GLOBAL_BATCH_SIZE below then,

python calculate_reference_lr.py \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --data_type instruction_tuning \
    --per_device_train_batch_size 1 \
    --max_seq_len 2048 --padding False --train_on_prompt True \
    --cache_dir ../default_dataset_cache --output_dir outputs
"""

GLOBAL_BATCH_SIZE = 128

BASE_LR = 3e-4
BASE_BS = 1 << 22  # 4M tokens according to LLaMA paper


def main():
    args = parse_args(verbose=False)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    tokenizer = build_tokenizer(args)
    train_dataset = build_dataset(args, tokenizer)

    # Prepare data_collator and eval_dataset
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           padding=True,
                                           pad_to_multiple_of=32,
                                           label_pad_token_id=IGNORE_INDEX,
                                           return_tensors='pt')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.per_device_train_batch_size,
                                  shuffle=False,
                                  collate_fn=data_collator)
    valid_tokens, total_tokens = 0, 0

    for batch in train_dataloader:
        valid_tokens += batch['labels'].ne(IGNORE_INDEX).sum().item()
        total_tokens += batch['labels'].numel()

    valid_ratio = valid_tokens / total_tokens
    valid_tokens_pre_batch = valid_tokens / math.ceil(
        len(train_dataset) / GLOBAL_BATCH_SIZE)
    reference_lr = BASE_LR * math.sqrt(
        valid_tokens_pre_batch / BASE_BS)  # lr ~ sqrt(batch_size)

    print(f"Valid Token Ratio: {valid_ratio * 100:.2f}%")
    print(
        f"Global Batch Size: {GLOBAL_BATCH_SIZE} | Effective Batch Size: {valid_tokens_pre_batch:.2f}"
    )

    print()
    print('-' * 20 + "Model Other Than Mistral/Gemma" + '-' * 20)
    print(f" * Reference LR (<30B)      : {reference_lr:.2e}")
    print(f" * Reference LR (30B-70B)   : {reference_lr / 2:.2e}")

    print()
    print('-' * 20 + "Model Mistral/Gemma" + '-' * 20)
    print(f" * Reference LR (<30B)      : {reference_lr / 6:.2e}")
    print(f" * Reference LR (30B-70B)   : {reference_lr / 12:.2e}")


if __name__ == '__main__':
    main()
