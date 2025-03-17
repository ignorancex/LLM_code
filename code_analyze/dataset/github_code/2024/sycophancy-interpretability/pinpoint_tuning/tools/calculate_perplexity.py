#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:55
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:12
Description        : Brought and modified from LLaMA-Factory.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import sys

sys.path.append('..')

import json
import os

import deepspeed
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from dataset import build_dataset
from model import build_model, build_tokenizer
from utils import IGNORE_INDEX
from utils.arguments import parse_args
"""
EXAMPLE:
python calculate_perplexity.py \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --data_type instruction_tuning \
    --per_device_train_batch_size 4 \
    --max_seq_len 2048 --padding False --train_on_prompt True \
    --training False \
    --cache_dir ../default_dataset_cache --output_dir outputs
"""


def main():
    args = parse_args(verbose=False)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    model = build_model(args)
    tokenizer = build_tokenizer(args)
    train_dataset = build_dataset(args, tokenizer)

    # Prepare data_collator and eval_dataset
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           padding=True,
                                           pad_to_multiple_of=32,
                                           label_pad_token_id=IGNORE_INDEX,
                                           return_tensors='pt')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.per_device_train_batch_size,
                                  shuffle=False,
                                  collate_fn=data_collator,
                                  pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    total_ppl = 0
    perplexities = []

    with torch.no_grad():
        pbar = tqdm(train_dataloader)
        for batch in pbar:

            batch = batch.to(model.device)
            outputs = model(input_ids=batch["input_ids"],
                            labels=batch["labels"],
                            attention_mask=batch["attention_mask"])

            shifted_logits = outputs.logits[..., :-1, :].contiguous()
            shifted_labels = batch["labels"][..., 1:].contiguous()

            loss_mask = shifted_labels.ne(IGNORE_INDEX)

            if loss_mask.sum() == 0:
                continue

            flattened_logits = shifted_logits.view(
                shifted_labels.size(0) * shifted_labels.size(1), -1)
            flattened_labels = shifted_labels.view(-1)

            token_logprob = criterion(flattened_logits, flattened_labels)
            token_logprob = token_logprob.view(shifted_logits.size(0), -1)

            sentence_logprob = (token_logprob *
                                loss_mask).sum(-1) / loss_mask.sum(-1)

            total_ppl += sentence_logprob.exp().sum().item()
            perplexities.extend(sentence_logprob.exp().tolist())

            pbar.set_description(
                f"AVG PPL: {sum(perplexities) / len(perplexities):.4f}")
            pbar.update(1)

        log_item = {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "perplexities": total_ppl / len(perplexities),
        }

        with open('perplexities.jsonl', 'a', encoding='utf-8') as file:
            file.write(json.dumps(log_item, ensure_ascii=False) + '\n')

        print("Average perplexity is {:.4f}".format(total_ppl /
                                                    len(perplexities)))
        print("Perplexities have been saved at {}.".format(
            'tools/perplexities.jsonl'))


if __name__ == '__main__':
    main()
