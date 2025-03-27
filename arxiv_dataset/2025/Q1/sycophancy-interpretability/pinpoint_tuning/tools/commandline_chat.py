#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-12 11:03
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:18
Description        : Chat with an instruction-tuned model.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import argparse

import torch
from peft import PeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
"""
EXAMPLE: CUDA_VISIBLE_DEVICES=0 python commandline_chat.py --model_path /path/to/model --peft_path /path/to/lora
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default='Qwen2.5-0.5B-Instruct')
    parser.add_argument('--peft_path', type=str, default=None)
    args = parser.parse_args()

    print("Initializing model and tokenizer ...\nPlease wait ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.float16,
                                                 device_map='auto')

    if args.peft_path is not None:
        model = PeftModelForCausalLM.from_pretrained(model, args.peft_path)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
    ]

    while True:
        print('-' * 30)
        prompt = input('>>> ')
        print()

        if prompt.lower() == 'exit':
            break

        if prompt.lower() == 'clear':
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
            ]
            continue

        messages.append({"role": "user", "content": prompt})

        inputs = tokenizer.apply_chat_template(messages,
                                               tokenize=True,
                                               add_generation_prompt=True,
                                               return_tensors="pt").to(
                                                   model.device)
        outputs = model.generate(inputs,
                                 max_new_tokens=256,
                                 do_sample=True,
                                 temperature=0.9,
                                 top_p=0.7)
        output_text = tokenizer.decode(outputs[0][inputs.size(1):],
                                       skip_special_tokens=True)
        print(output_text)

        messages.append({"role": "assistant", "content": output_text})


if __name__ == '__main__':
    main()
