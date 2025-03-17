#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-10 17:24
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-12 22:59
Description        : 
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import sys

sys.path.append('..')

import argparse
import json

from transformers import AutoTokenizer

from dataset.dataset_json import JsonDataset
from utils import IGNORE_INDEX
"""
python test_dataset_json.py --model_path model_examples/Qwen2.5-7B-Instruct
"""

parser = argparse.ArgumentParser(description="Test JSON dataset class.")
parser.add_argument("--model_path",
                    type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")

args = parser.parse_args()
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("#" * 50)
print("Test Instruction Tuning Dataset...")

data_path = 'dataset_examples/instruction_tuning_example.jsonl'
with open(data_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print(" - Testing padding=True, train_on_prompt=True")
tokenized_data = JsonDataset._tokenize_data(
    data,
    tokenizer,
    max_seq_length=256,
    padding=True,
    data_type='instruction_tuning',
    num_workers=4,
    train_on_prompt=True,
)

print(" -*- Tokenized Labels -*-")
print(tokenized_data[0]['labels'])
print(" -*- Decoded Labels -*-")
labels = [
    label if label != IGNORE_INDEX else tokenizer.encode('@')[0]
    for label in tokenized_data[0]['labels']
]
print(tokenizer.decode(labels))
print("-" * 50)

print(" - Testing padding=True, train_on_prompt=False")
tokenized_data = JsonDataset._tokenize_data(
    data,
    tokenizer,
    max_seq_length=256,
    padding=True,
    data_type='instruction_tuning',
    num_workers=4,
    train_on_prompt=False,
)

print(" -*- Tokenized Labels -*-")
print(tokenized_data[-1]['labels'])
print(" -*- Decoded Labels -*-")
labels = [
    label if label != IGNORE_INDEX else tokenizer.encode('@')[-1]
    for label in tokenized_data[-1]['labels']
]
print(tokenizer.decode(labels))
print("-" * 50)

print(" - Testing padding=False, train_on_prompt=False")
tokenized_data = JsonDataset._tokenize_data(
    data,
    tokenizer,
    max_seq_length=256,
    padding=False,
    data_type='instruction_tuning',
    num_workers=4,
    train_on_prompt=False,
)

print(" -*- Tokenized Labels -*-")
print(tokenized_data[-1]['labels'])
print(" -*- Decoded Labels -*-")
labels = [
    label if label != IGNORE_INDEX else tokenizer.encode('@')[-1]
    for label in tokenized_data[-1]['labels']
]
print(tokenizer.decode(labels))
print("-" * 50)

print("#" * 50)
print("Test Pretraining Dataset...")

data_path = 'dataset_examples/pretraining_example.jsonl'
with open(data_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print(" - Testing padding=True")
tokenized_data = JsonDataset._tokenize_data(
    data,
    tokenizer,
    max_seq_length=256,
    padding=True,
    data_type='pretraining',
    num_workers=4,
)

print(" -*- Tokenized Labels -*-")
print(tokenized_data[0]['labels'])
print(" -*- Decoded Labels -*-")
labels = [
    label if label != IGNORE_INDEX else tokenizer.encode('@')[0]
    for label in tokenized_data[0]['labels']
]
print(tokenizer.decode(labels))
print("-" * 50)

print(" - Testing padding=False")
tokenized_data = JsonDataset._tokenize_data(
    data,
    tokenizer,
    max_seq_length=256,
    padding=False,
    data_type='pretraining',
    num_workers=4,
)

print(" -*- Tokenized Labels -*-")
print(tokenized_data[0]['labels'])
print(" -*- Decoded Labels -*-")
labels = [
    label if label != IGNORE_INDEX else tokenizer.encode('@')[0]
    for label in tokenized_data[0]['labels']
]
print(tokenizer.decode(labels))
print("-" * 50)
