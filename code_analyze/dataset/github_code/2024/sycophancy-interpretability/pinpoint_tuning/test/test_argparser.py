#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-11 18:24
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-12 16:48
Description        : 
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import sys

sys.path.append('..')

import deepspeed

from utils import transformers_logging_setup
from utils.arguments import parse_args
"""
python test_argparser.py --output_dir 123 --model_path model_examples/Qwen2.5-7B-Instruct --peft_type None --padding True
"""

transformers_logging_setup()
args = parse_args()

assert args.peft_type == None, "peft_type must be None but not str-object 'None'."
assert args.padding == True, "padding must be True but but not str-object 'True'."
