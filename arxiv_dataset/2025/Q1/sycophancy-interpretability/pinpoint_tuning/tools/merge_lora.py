#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:29
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:19
Description        : Merge LoRA model into base model.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import argparse
import os
import shutil

import torch
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
"""
EXAMPLE: CUDA_VISIBLE_DEVICES=0 python merge_lora.py --model_path /path/to/base_model --lora_path /path/to/lora_model --output_path /path/to/merged_model
"""


def main():
    parser = argparse.ArgumentParser(
        description='Merge LoRA model into base model.')

    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to the base model.')
    parser.add_argument('--lora_path',
                        type=str,
                        required=True,
                        help='Path to the LoRA model.')
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='Path to save the merged model.')

    args = parser.parse_args()
    model_path = args.model_path
    lora_path = args.lora_path
    output_path = args.output_path

    print(f'Loading base model from {model_path} ...')
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    print(f'Loading LoRA model from {lora_path} ...')
    lora_model = PeftModelForCausalLM.from_pretrained(
        model, lora_path, torch_dtype=torch.float16)

    print('Merging models ...')
    merged_model = lora_model.merge_and_unload()

    print(f'Saving merged model to {output_path} ...')
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # In case the model is not from huggingface, we need to copy the model file (*.py) to the output directory
    for file in os.listdir(model_path):
        if file.endswith('.py'):
            shutil.copy(os.path.join(model_path, file), output_path)


if __name__ == '__main__':
    main()
