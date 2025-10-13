# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

# CUDA_VISIBLE_DEVICES=1 python llama_generate_answer.py

import sys
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from fastchat.model import get_conversation_template
model_dir = '../../pretrained_models/Qwen-14B-Chat'
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.eos_token_id = 151643
tokenizer.pad_token = tokenizer.eos_token


def get_data(src_dir):
    print('data path:', src_dir)
    lines = open(src_dir).readlines()
    prompts = []

    for d in lines:
        d = json.loads(d)
        prompts.append(d['prompt'])
    print('prompt num', len(prompts))
    return prompts

dataset_name = 'amz-new'
framework = 'KAR'
file_name = 'recent_history'
prompts = get_data(src_dir=f'../data/{dataset_name}/proc_data/{framework}/{file_name}.json')
for i in tqdm(range(len(prompts))):
    prompt = prompts[i]
    response, history = model.chat(tokenizer, prompt, history=None)
    if i == 0:
        print(prompt)
        print(response)