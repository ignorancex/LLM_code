#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-10 20:40
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-12 16:48
Description        : 
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import sys

sys.path.append('..')

import argparse

from transformers import AutoTokenizer

from model import build_tokenizer_hf

parser = argparse.ArgumentParser(description="Test JSON dataset class.")
parser.add_argument("--model-path",
                    type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")

args = parser.parse_args()
tokenizer = build_tokenizer_hf(args)

print(tokenizer.encode("Hello world"))
