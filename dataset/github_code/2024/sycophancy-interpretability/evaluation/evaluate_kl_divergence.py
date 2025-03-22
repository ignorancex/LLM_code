#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-17 21:53
Last Modified By   : 殊理 (husile.hsl@alibaba-inc.com)
Last Modified Date : 2024-10-18 14:40
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import argparse
import json

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from transformers import set_seed

from utils import build_model_tokenizer_hf

"""
python evaluate_kl_divergence.py \
    --model_path_0 /path/to/model_0 \
    --model_path_1 /path/to/model_1
"""


def main():
    parser = argparse.ArgumentParser(description='This is the program to evaluate the models on translation task.')
    parser.add_argument('--model_path_0', type=str, default=None, help='Model type')
    parser.add_argument('--model_path_1', type=str, default=None, help='Model type')
    parser.add_argument('-s',  '--seed', type=int, default=0)

    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    # 读取数据文件，并随机采样其中的若干条数据
    num_samples = 1000
    all_data = []
    with open('datasets/openwebtext-10k.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            all_data.append(json.loads(line))

    # 随机选择其中的一个子集使用，因为设置了随机种子所以应该不同次之间是一致的
    rand_idxs = np.random.choice(len(all_data), num_samples, replace=False)

    model_path_0 = args.model_path_0
    model_path_1 = args.model_path_1

    #设置gpu设备
    device = 0
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device_0 = torch.device(device)

    device = 1
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device_1 = torch.device(device)

    model_0, tokenizer = build_model_tokenizer_hf(model_path_0, device=device_0)
    model_1, _         = build_model_tokenizer_hf(model_path_1, device=device_1)

    kl_divs = []

    with torch.no_grad():
        for i in tqdm(rand_idxs):
            # Select the first 128 tokens
            input_ids = tokenizer(all_data[i]['text'], return_tensors='pt')['input_ids'][:, :128]
            
            input_ids = input_ids.to(model_0.device)
            orig_logits = model_0(input_ids).logits.cpu().type(torch.float32)
            orig_probs = F.softmax(orig_logits, dim=-1)

            input_ids = input_ids.to(model_1.device)
            logits = model_1(input_ids).logits.cpu().type(torch.float32)
            probs = F.softmax(logits, dim=-1)

            kl_div = (orig_probs * (orig_probs / probs).log()).sum() / (input_ids.shape[-1] * input_ids.shape[-2])
            kl_divs.append(kl_div.item())

    print(f"Mean KL divergence: {np.mean(kl_divs)}.")


if __name__ == '__main__':
    main()