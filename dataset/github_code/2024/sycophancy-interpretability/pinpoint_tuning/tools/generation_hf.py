#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:17
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:12
Description        : An example of using Huggingface transformers to generate text.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import argparse
import json
import os
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
"""
EXAMPLE:
python generation_hf.py --model_path /path/to/model --data_path /path/to/data --output_dir /path/to/output
"""


def read_jsonl(path: str) -> List[dict]:
    return [json.loads(line) for line in open(path, 'r', encoding='utf-8')]


def build_model_tokenizer(
        model_path: str,
        torch_dtype: torch.Type) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map='auto',
                                                 torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def filter_already_generated(data: list, output_path: str,
                             _field: str) -> list:
    generated_data = read_jsonl(output_path)
    generated_questions = [datapoint[_field] for datapoint in generated_data]
    return [
        datapoint for datapoint in data
        if datapoint[_field] not in generated_questions
    ]


def main():
    parser = argparse.ArgumentParser(
        description=
        "Example of using Huggingface transformers to generate text.")

    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to the model.')
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Path to the data.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='generation_results/results_hf',
                        help='Path to save the output.')
    parser.add_argument('--torch_dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    torch_dtype = getattr(torch, args.torch_dtype)
    batch_size = args.batch_size

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir,
                               os.path.basename(model_path) + '.jsonl')

    # Prepare model, tokenizer and generation configuration
    model, tokenizer = build_model_tokenizer(model_path, torch_dtype)
    model.generation_config = GenerationConfig.from_pretrained(
        model_path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True)
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.0

    all_data = read_jsonl(data_path)

    # Filter already generated data
    if os.path.exists(output_path):
        print(
            f"Found existing output file {output_path}, skipping already generated data."
        )
        all_data = filter_already_generated(all_data, output_path, 'question')

    for index in tqdm(range(0, len(all_data), batch_size)):
        batch_data = all_data[index:index + batch_size]

        input_texts = []
        for datapoint in batch_data:
            dialogue = [{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": datapoint['question']
            }]
            input_texts.append(
                tokenizer.apply_chat_template(dialogue,
                                              tokenize=False,
                                              add_generation_prompt=True))

        completions = []
        for input_text in input_texts:
            inputs = tokenizer(input_text,
                               return_tensors='pt').to(model.device)
            output_ids = model.generate(
                **inputs,
                generation_config=model.generation_config,
                max_new_tokens=1024)
            completion = tokenizer.decode(
                output_ids[0][inputs.input_ids.size(1):],
                skip_special_tokens=True)
            completions.append(completion)

        # Save batch results
        with open(output_path, 'a', encoding='utf-8') as file:
            for datapoint, completion in zip(batch_data, completions):
                datapoint['completion'] = completion
                file.write(json.dumps(datapoint, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
