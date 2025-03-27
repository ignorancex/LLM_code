#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-17 17:31
Last Modified By   : 殊理 (husile.hsl@alibaba-inc.com)
Last Modified Date : 2024-10-18 14:20
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import argparse
import json
import os
from tqdm import tqdm
from vllm import SamplingParams

from utils import EOS_TOKEN_LIST, HUMAN_INSTRUCTIONS
from utils import build_model_tokenizer, construct_text_from_messages, filter_already_generated, read_jsonl

"""
python evaluate_sycophancy_chat_vllm.py \
    --model_path /path/to/model \
    --data_path /path/to/data
"""

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
                        default='results/sycophancy_eval',
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

    model_path = args.model_path
    data_path = args.data_path
    torch_dtype = args.torch_dtype
    batch_size = args.batch_size

    model_name = os.path.basename(model_path.rstrip('/'))
    data_name = os.path.basename(data_path.rstrip('/')).split('.')[-2]

    output_dir = os.path.join(args.output_dir, data_name, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare model, tokenizer and generation configuration
    model, tokenizer = build_model_tokenizer(model_path, torch_dtype,
                                             args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0.,
                                     top_p=.95,
                                     max_tokens=1024,
                                     stop=EOS_TOKEN_LIST +
                                     [tokenizer.eos_token])
    
    # We need to test sycophancy in multi-round conversations.
    #  - First, the LLM answers the question.
    #  - Second, the user challenges the LLM.
    #  - Third, the LLM answers the question again.
    # A ideal LLM should be able to keep the correct answer while modifying the false answer since
    #   previous works have shown that the truthfulness inside LLMs is much higher than it actually behaves.
    for round_index, instruction in enumerate(HUMAN_INSTRUCTIONS[data_name]):
        
        if round_index != 0:
            data_path = os.path.join(output_dir, f'round_{round_index}.jsonl')

        all_data = read_jsonl(data_path)

        output_path = os.path.join(output_dir, f'round_{round_index + 1}.jsonl')

        # Filter already generated data
        if os.path.exists(output_path):
            print(
                f"Found existing output file {output_path}, skipping already generated data."
            )
            all_data, _ = filter_already_generated(all_data, output_path, 'id')

        if len(all_data) > 0:
            print(construct_text_from_messages(all_data[0]['prompt'], tokenizer, continue_generation=True))

        for index in tqdm(range(0, len(all_data), batch_size)):
            batch_data = all_data[index:index + batch_size]

            input_texts = []
            for datapoint in batch_data:
                input_texts.append(construct_text_from_messages(datapoint['prompt'], tokenizer, continue_generation=True))

            if index == 0:
                print(input_texts[0])

            outputs = model.generate(input_texts, sampling_params=sampling_params)
            completions = [output.outputs[0].text for output in outputs]

            # Save batch results
            with open(output_path, 'a', encoding='utf-8') as file:
                for datapoint, completion in zip(batch_data, completions):
                    if datapoint['prompt'][-1]['role'] == 'assistant':
                        datapoint['prompt'][-1]['content'] += completion
                    else:
                        datapoint['prompt'].append({"role": "assistant", "content": completion})
                    if instruction is not None:
                        datapoint['prompt'].append({"role": "user", "content": instruction})
                    file.write(json.dumps(datapoint, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
