#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:30
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-18 19:23
Description        : Main script for training.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import logging

from transformers import set_seed

from dataset import build_dataset
from model import build_model, build_tokenizer
from trainer import build_trainer
from utils import logging_rank, transformers_logging_setup
from utils.arguments import parse_args

from utils.utils_spt import freeze_modules, freeze_lora_modules

logger = logging.getLogger('DeepSpeed')


def main():

    transformers_logging_setup()
    args = parse_args()

    # Set seed before initializing model.
    set_seed(args.seed)

    model = build_model(args)
    tokenizer = build_tokenizer(args)

    logging_rank("tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id))
    logging_rank("tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id))
    logging_rank("tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id))

    if args.precise_level > 0 and args.path_patching_path is not None:
        if args.model_type in ['qwen2', 'llama', 'mistral']:
            if args.peft_type is not None and args.peft_type == 'lora':
                freeze_lora_modules(model, args.path_patching_path, args.precise_level, args.train_topk, args.train_kv)
            else:
                freeze_modules(model, args.path_patching_path, args.precise_level, args.train_topk, args.train_kv)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}. You need to implement freeze_modules for your model type.")

    train_data = build_dataset(args, tokenizer)
    logging_rank(f"length of train dataset: {len(train_data)}")

    trainer = build_trainer(args, model, tokenizer, train_data)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir, keep_zero=True)

    logger.info("-*-" * 25 + "\nTraining completed! Congratulations!")


if __name__ == '__main__':
    main()
