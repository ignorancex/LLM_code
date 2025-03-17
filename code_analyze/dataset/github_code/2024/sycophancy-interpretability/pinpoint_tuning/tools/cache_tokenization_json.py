#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:29
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:18
Description        : Cache tokenization jsonl data.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import sys

sys.path.append('..')

import os

import deepspeed

from dataset import build_dataset
from model import build_tokenizer
from utils.arguments import parse_args
"""
EXAMPLE: 
python cache_tokenization_json.py \
    --model_path /path/to/tokenizer \
    --data_path /path/to/data \
    --data_type instruction_tuning \
    --max_seq_len 2048 --padding False --train_on_prompt True \
    --cache_dir ../default_dataset_cache --output_dir outputs
"""


def main():
    args = parse_args(verbose=False)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    tokenizer = build_tokenizer(args)
    _ = build_dataset(args, tokenizer)


if __name__ == '__main__':
    main()
