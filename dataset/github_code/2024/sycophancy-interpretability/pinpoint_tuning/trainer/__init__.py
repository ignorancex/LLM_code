#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-10 22:42
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:19
Description        : __init__.py
-------- 
Copyright (c) 2024 Wei Chen. 
'''

from argparse import Namespace

from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer, DataCollatorForSeq2Seq

from trainer.trainer_hf import build_trainer_hf
from utils import logging_rank
from utils import IGNORE_INDEX


def build_trainer(args: Namespace, model: PreTrainedModel,
                  tokenizer: PreTrainedTokenizer,
                  train_dataset: Dataset) -> Trainer:
    """Build trainer based on args.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    model : PreTrainedModel
        Training model.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to tokenize data.
    train_dataset : Dataset
        Tokenized dataset.

    Returns
    -------
    Trainer
        Training trainer.
    """

    # Prepare data_collator and eval_dataset
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           padding=True,
                                           pad_to_multiple_of=32,
                                           label_pad_token_id=IGNORE_INDEX,
                                           return_tensors='pt')
    logging_rank(f"data collator: {data_collator}")

    trainer = build_trainer_hf(args, model, tokenizer, train_dataset,
                               data_collator)
    return trainer
