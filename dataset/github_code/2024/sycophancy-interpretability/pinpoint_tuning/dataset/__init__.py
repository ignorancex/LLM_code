#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-10 13:44
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:17
Description        : __init__.py
-------- 
Copyright (c) 2024 Wei Chen. 
'''

from argparse import Namespace

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from dataset.dataset_json import build_dataset_json


def build_dataset(args: Namespace, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Build dataset according to args.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to tokenize data.

    Returns
    -------
    Dataset
        Tokenized dataset.
    """

    dataset = build_dataset_json(args, tokenizer)
    return dataset
