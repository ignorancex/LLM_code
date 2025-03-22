#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:55
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:12
Description        : Brought and modified from LLaMA-Factory.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import sys

sys.path.append("..")

import deepspeed
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile

from model import build_model
from utils.arguments import parse_args
"""
EXAMPLE: 
python calculate_flops.py \
    --model_path /path/to/model \
    --attn_implementation sdpa \
    --per_device_train_batch_size 1 \
    --max_seq_len 1024 \
    --training False \
    --output_dir outputs
"""


def main():
    args = parse_args(verbose=False)

    model = build_model(args)

    fake_input = torch.ones(
        (args.per_device_train_batch_size, args.max_seq_length),
        dtype=torch.long).to(model.device)
    input_dict = {"input_ids": fake_input, "labels": fake_input.clone()}

    flops, macs, params = get_model_profile(model=model,
                                            kwargs=input_dict,
                                            print_profile=True,
                                            detailed=True)

    print("-" * 30)
    print(f"FLOPs   : {flops}")
    print(f"MACs    : {macs}")
    print(f"Params  : {params}")


if __name__ == "__main__":
    main()
