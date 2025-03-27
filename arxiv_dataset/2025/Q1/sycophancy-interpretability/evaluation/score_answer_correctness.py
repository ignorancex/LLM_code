#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-17 21:51
Last Modified By   : 殊理 (husile.hsl@alibaba-inc.com)
Last Modified Date : 2024-10-18 14:47
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import argparse
import json
import os
from typing import Dict

from tqdm import tqdm
from vllm import SamplingParams

from utils import EOS_TOKEN_LIST, CORRECTNESS_EVAL_TEMPLATE
from utils import build_model_tokenizer, construct_text_from_messages, filter_already_generated, read_jsonl

"""
python score_answer_correctness.py \
    --model_path /path/to/model \
    --eval_model_path /path/to/eval_model \
    --results_dir /path/to/results
"""

ROUND_LIST = [1, 3] # Rounds to be evaluated, for free generation, this should be [1, 3]

def construct_messages(datapoint: Dict) -> Dict:

    question = datapoint['base']['question']
    if datapoint['prompt'][-1]['role'] == 'user':
        response = datapoint['prompt'][-2]['content']
    else:
        response = datapoint['prompt'][-1]['content']
    answer = datapoint['base']['correct_answer']

    prompt = CORRECTNESS_EVAL_TEMPLATE.format(question=question, response=response, answer=answer)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


def main():
    parser = argparse.ArgumentParser(
        description=
        "Example of using Huggingface transformers to generate text.")

    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to the model.')
    parser.add_argument('--eval_model_path',
                        type=str,
                        default='Qwen/Qwen2.5-72B-Instruct',
                        help='Path to the model used to evaluate.')
    parser.add_argument('--results_dir',
                        type=str,
                        required=True,
                        help='Path to the data.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='results/sycophancy_eval_correctness',
                        help='Path to save the output.')
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

    model_path = args.model_path.rstrip('/')
    eval_model_path = args.eval_model_path.rstrip('/')
    results_dir = args.results_dir.rstrip('/')
    torch_dtype = args.torch_dtype
    batch_size = args.batch_size

    model_name = os.path.basename(model_path)
    data_name = os.path.basename(results_dir)

    output_dir = os.path.join(args.output_dir, data_name, os.path.basename(model_name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare model, tokenizer and generation configuration
    model, tokenizer = build_model_tokenizer(eval_model_path, torch_dtype,
                                             args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0.,
                                     top_p=.95,
                                     max_tokens=16,
                                     stop=EOS_TOKEN_LIST +
                                     [tokenizer.eos_token])

    for round_index in ROUND_LIST:
        data_path = os.path.join(results_dir, model_name, f'round_{round_index}.jsonl')
        all_data = read_jsonl(data_path)

        output_path = os.path.join(output_dir, f'round_{round_index}.jsonl')

        # Filter already generated data
        if os.path.exists(output_path):
            print(
                f"Found existing output file {output_path}, skipping already generated data."
            )
            all_data, _ = filter_already_generated(all_data, output_path, 'id')

        for index in tqdm(range(0, len(all_data), batch_size)):
            batch_data = all_data[index:index + batch_size]

            input_texts = []
            for datapoint in batch_data:
                messages = construct_messages(datapoint)
                input_texts.append(construct_text_from_messages(messages, tokenizer, continue_generation=True))

            if index == 0:
                print(input_texts[0])

            outputs = model.generate(input_texts, sampling_params=sampling_params)
            completions = [output.outputs[0].text for output in outputs]

            # Save batch results
            with open(output_path, 'a', encoding='utf-8') as file:
                for datapoint, completion in zip(batch_data, completions):
                    item = {
                        'id': datapoint['id'],
                        'is_correct': completion
                    }
                    file.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
