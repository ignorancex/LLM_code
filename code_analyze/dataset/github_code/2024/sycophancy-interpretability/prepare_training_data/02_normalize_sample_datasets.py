#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-15 18:37
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-15 20:44
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import copy
import json
import os
import random
import re

from datasets import load_dataset

random.seed(42)

# Default format for each datapoint.
default_data_obj = {
    "question": "",
    "options": [],
    "correct_answer": "",
    "incorrect_answer": "",
    "candidate_answers": [],
    "rationale": "",
    "explanation": "",
    "info": {
        "dataset": "",
        "id": -1,
    }
}

OUTPUT_DIR = 'datasets_sampled'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# MMLU

dataset_name = 'mmlu'
dataset_path = './datasets/mmlu'

dataset = load_dataset(dataset_path, 'auxiliary_train')['train']

all_data = []

for cnt, item in enumerate(dataset):
    item = item['train']
    data_obj = copy.deepcopy(default_data_obj)
    
    data_obj['question'] = item['question']
    data_obj['options'] = item['choices']

    data_obj['correct_answer'] = chr(65 + item['answer'])
    data_obj['incorrect_answer'] = chr(65 + random.choice([i for i in range(len(data_obj['options'])) if i != item['answer']]))

    data_obj['info']['dataset'] = dataset_name
    data_obj['info']['id'] = cnt

    all_data.append(data_obj)

print(json.dumps(all_data[0], indent=4))

print(f"Finish processing {dataset_name} dataset...")
print(f"Total number of data: {len(all_data)}")

sampled_data = random.sample(all_data, 20000)

with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_sampled_20k.jsonl"), "w", encoding="utf-8") as file:
    for item in sampled_data:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# MATH-QA

dataset_name = 'math'
dataset_path = './datasets/mathqa/train.json'

pattern = r'^[a-zA-Z]\s*\)?\s*(.*)'

with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

all_data = []

for cnt, item in enumerate(dataset):
    data_obj = copy.deepcopy(default_data_obj)
    data_obj['question'] = item["Problem"]
    
    if item['options'][0] == '[':
        item['options'] = eval(item['options'])
    
    if isinstance(item['options'], str):
        all_options = item["options"].split(' , ')
    else:
        all_options = item['options']
    
    skip = False
    for option in all_options:
        match = re.match(pattern, option.strip())
        if match:
            data_obj['options'].append(match.group(1))
        else:
            skip = True
            break
    
    if skip:
        continue

    data_obj['correct_answer'] = item["correct"].upper()
    data_obj['incorrect_answer'] = chr(65 + random.choice([i for i in range(len(data_obj['options'])) if i != ord(data_obj['correct_answer']) - 65]))
    data_obj['rationale'] = item['Rationale']

    data_obj['info']['dataset'] = dataset_name
    data_obj['info']['id'] = cnt

    all_data.append(data_obj)
    cnt += 1

print(json.dumps(all_data[0], indent=4))

print(f"Finish processing {dataset_name} dataset...")
print(f"Total number of data: {len(all_data)}")

sampled_data = random.sample(all_data, 20000)

with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_sampled_20k.jsonl"), "w", encoding="utf-8") as file:
    for item in sampled_data:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# AQUA

dataset_name = 'aqua'
dataset_path = './datasets/aqua/train.json'

pattern = r'^[a-zA-Z]\s*\)?\s*(.*)'

dataset = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            cur_line = json.loads(line)
            dataset.append(cur_line)
        except:
            print(line)
            continue

all_data = []

for cnt, item in enumerate(dataset):
    data_obj = copy.deepcopy(default_data_obj)
    data_obj['question'] = item['question']

    for option in item['options']:
        match = re.match(pattern, option.strip())
        if match:
            data_obj['options'].append(match.group(1))
        else:
            assert False, option

    data_obj['correct_answer'] = item["correct"].upper()
    data_obj['incorrect_answer'] = chr(65 + random.choice([i for i in range(len(data_obj['options'])) if i != ord(data_obj['correct_answer']) - 65]))
    data_obj['rationale'] = item['rationale']

    data_obj['info']['dataset'] = dataset_name
    data_obj['info']['id'] = cnt

    all_data.append(data_obj)

print(json.dumps(all_data[0], indent=4))

print(f"Finish processing {dataset_name} dataset...")
print(f"Total number of data: {len(all_data)}")

sampled_data = random.sample(all_data, 20000)

with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_sampled_20k.jsonl"), "w", encoding="utf-8") as file:
    for item in sampled_data:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# TriviaQA

dataset_name = "trivia"
dataset_path = "datasets/trivia_qa/unfiltered.nocontext"

dataset = load_dataset(dataset_path, split=['train'])[0]

all_data = []

for cnt, item in enumerate(dataset):
    data_obj = copy.deepcopy(default_data_obj)
    data_obj['question'] = item['question']

    data_obj['correct_answer'] = item["answer"]["value"]
    data_obj['candidate_answers'] = item["answer"]["aliases"]

    data_obj['info']['dataset'] = dataset_name
    data_obj['info']['id'] = cnt

    all_data.append(data_obj)

print(json.dumps(all_data[0], indent=4))

print(f"Finish processing {dataset_name} dataset...")
print(f"Total number of data: {len(all_data)}")

sampled_data = random.sample(all_data, 20000)

with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_sampled_20k_no_incorrect.jsonl"), "w", encoding="utf-8") as file:
    for item in sampled_data:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
