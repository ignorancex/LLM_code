#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-16 23:14
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-18 13:27
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

from transformers import AutoTokenizer

from utils import read_jsonl, construct_text_from_messages

model_path = 'Qwen/Qwen2.5-7B-Instruct'
# model_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
# model_path = 'mistralai/Mistral-7B-Instruct-v0.2'

tokenizer = AutoTokenizer.from_pretrained(model_path)

print('-' * 50)

dataset = read_jsonl('datasets/sycophancy_eval/multiple_choice.jsonl')

print(dataset[0])

text = construct_text_from_messages(dataset[0]['prompt'], tokenizer, continue_generation=True)
print(text)

print('-' * 50)

dataset = read_jsonl('datasets/sycophancy_eval/multiple_choice_cot.jsonl')

print(dataset[0])

text = construct_text_from_messages(dataset[0]['prompt'], tokenizer, continue_generation=True)
print(text)

print('-' * 50)

dataset = read_jsonl('datasets/sycophancy_eval/free_generation.jsonl')

print(dataset[0])

text = construct_text_from_messages(dataset[0]['prompt'], tokenizer, continue_generation=True)
print(text)
