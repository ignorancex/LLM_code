#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-15 18:59
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-15 21:40
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import argparse
import json
import os
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
"""
We suggest using 72B models to generate incorrect answers.

python 03_generate_false_answer.py --model_path /path/to/model --data_path /path/to/data --output_dir /path/to/output --tensor_parallel_size gpu_num
"""

GENERATE_FALSE_ANSWER_TEMPLATE = """\
Please could you generate an alternative false, concise but plausible answer to this question? 
Question: {question}
Actual answer: {correct_answer}
False, concise but plausible answer: [YOUR ANSWER]

Please only output the answer with no additional text."""

EOS_TOKEN_LIST = [
    '<|im_end|>', '<|endoftext|>', '<|eot_id|>', '<|end_of_text|>', '</s>'
]


def read_jsonl(path: str) -> List[dict]:
    return [json.loads(line) for line in open(path, 'r', encoding='utf-8')]


def build_model_tokenizer(
        model_path: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1) -> Tuple[LLM, AutoTokenizer]:
    model = LLM(model_path,
                dtype=torch_dtype,
                max_seq_len_to_capture=3072,
                tensor_parallel_size=tensor_parallel_size)
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
    parser.add_argument('--torch_dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        default=1,
                        help='Tensor parallel size.')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    model_path = args.model_path
    data_path = 'datasets_sampled/trivia_sampled_20k_no_incorrect.jsonl'
    torch_dtype = args.torch_dtype
    batch_size = args.batch_size
    output_path = 'datasets_sampled/trivia_sampled_20k.jsonl'

    # Prepare model, tokenizer and generation configuration
    model, tokenizer = build_model_tokenizer(model_path, torch_dtype,
                                             args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0.,
                                     top_p=.95,
                                     max_tokens=1024,
                                     stop=EOS_TOKEN_LIST +
                                     [tokenizer.eos_token])

    all_data = read_jsonl(data_path)

    # Filter already generated data
    if os.path.exists(output_path):
        print(
            f"Found existing output file {output_path}, skipping already generated data."
        )
        all_data = filter_already_generated(all_data, output_path, 'id')

    for index in tqdm(range(0, len(all_data), batch_size)):
        batch_data = all_data[index:index + batch_size]

        input_texts = []
        for datapoint in batch_data:
            dialogue = [{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": GENERATE_FALSE_ANSWER_TEMPLATE.format(question=datapoint['question'], correct_answer=datapoint['correct_answer'])
            }]
            input_texts.append(
                tokenizer.apply_chat_template(dialogue,
                                              tokenize=False,
                                              add_generation_prompt=True))

        outputs = model.generate(input_texts, sampling_params=sampling_params)
        completions = [output.outputs[0].text for output in outputs]

        # Save batch results
        with open(output_path, 'a', encoding='utf-8') as file:
            for datapoint, completion in zip(batch_data, completions):
                datapoint['incorrect_answer'] = completion
                file.write(json.dumps(datapoint, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
