#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-17 21:53
Last Modified By   : 殊理 (husile.hsl@alibaba-inc.com)
Last Modified Date : 2024-10-18 15:48
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import build_model_tokenizer_hf, construct_text_from_messages

"""
python evaluate_mmlu.py \
    --model_path /path/to/model \
    --data_path datasets/mmlu/data \
    --batch_size 4
"""


def construct_messages(line):
    prompt = line["question"]
    for choice in CHOICES:
        prompt += f'\n({choice}) {line[f"{choice}"]}'

    prompt += "\nPlease answer with just the letter of correct answer."

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": "The correct answer is ("
        }
    ]

    return messages


TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
CHOICES = ["A", "B", "C", "D"]


def pad_and_construct_inputs(input_lists, padding_side='left', pad_token_id=0):
    """Pad input tokens to same length"""
    max_length = max([len(x) for x in input_lists])
    input_ids = torch.ones(len(input_lists), max_length, dtype=torch.long) * pad_token_id
    attention_mask = torch.zeros(len(input_lists), max_length, dtype=torch.long)

    for i, input_list in enumerate(input_lists):
        if padding_side == 'left':
            input_ids[i][max_length - len(input_list):] = torch.Tensor(input_list)
            attention_mask[i][max_length - len(input_list):] = 1
        elif padding_side == 'right':
            input_ids[i][:len(input_list)] = torch.Tensor(input_list)
            attention_mask[i][:len(input_list)] = 1
        else:
            assert False, "padding_side should be either left or right."

    return input_ids, attention_mask


def get_logits(tokenizer, model, inputs):

    input_lists = []
    for cur_input in inputs:
        input_tensor = tokenizer(cur_input, add_special_tokens=False, return_tensors="pt").input_ids
        input_lists.append(input_tensor[0])

    input_ids, _ = pad_and_construct_inputs(input_lists)
    input_ids = input_ids.to(model.device)

    # Cut off the input to 2048 tokens
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, input_ids.shape[1] - 2048 + 1 :]
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    outputs = model(input_ids, attention_mask=attention_mask)["logits"]
    logits = outputs[:, -1, :]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    save_result_dir=None,
    batch_size=1,
    **kwargs,
):
    result = []
    score = []

    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}

    test_tokenization = tokenizer.encode('(A', add_special_tokens=False)
    if len(test_tokenization) >= 2:
        choices_ids = torch.tensor(
            [tokenizer.encode('(' + item, add_special_tokens=False)[1] for item in CHOICES]
        ).unsqueeze(0).to(model.device)
    else:
        choices_ids = torch.tensor(
            [tokenizer.encode(item, add_special_tokens=False)[0] for item in CHOICES]
        ).unsqueeze(0).to(model.device)

    idx_list = list(range(0, len(test_df), batch_size))
    for i in tqdm(idx_list):
        input_texts = []
        answer_list = []
        for row in test_df.iloc[i:i+batch_size].to_dict(orient='records'):
            messages = construct_messages(row)
            input_texts.append(construct_text_from_messages(messages, tokenizer, continue_generation=True))
            if 'answer' in row:
                answer_list.append(row['answer'])

        logits = get_logits(tokenizer, model, input_texts)
        # print(torch.topk(logits[0], dim=-1, k=4))
        
        softval = logits.gather(1, choices_ids.expand(logits.size(0), -1)).softmax(1)
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()

        for i in range(len(probs)):
            for j, choice in enumerate(CHOICES):
                all_probs[f"prob_{choice}"].append(probs[i][j])
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs[i])]

            if answer_list != []:
                correct = 1 if pred == answer_list[i] else 0
                score.append(correct)
            result.append(pred)

    if save_result_dir:
        test_df["model_output"] = result
        for i, choice in enumerate(CHOICES):
            test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    score = torch.Tensor(score)
    print(torch.sum(score) / score.size(0))

    return score


def cal_mmlu(res):
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print("\n\n\n", "total cnt:", cnt, "\n")
    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print("%s ACC: %.2f " % (k, acc_sum_dict[k] / cnt_dict[k] * 100))
    
    print("AVERAGE ACC:%.2f " % (acc_sum / cnt * 100))

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
                        default='results/mmlu',
                        help='Path to save the output.')
    parser.add_argument('--torch_dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    model_path = args.model_path.rstrip('/')
    data_path = args.data_path
    torch_dtype = args.torch_dtype
    batch_size = args.batch_size

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir,
                               os.path.basename(model_path))

    # Prepare model, tokenizer and generation configuration
    model, tokenizer = build_model_tokenizer_hf(model_path, torch_dtype)

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):

        test_file_path = os.path.join(
            data_path, "test", f"{subject_name}_test.csv"
        )
        
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            save_result_dir=output_path,
            batch_size=batch_size
        )
        dev_result[subject_name] = score
    
    cal_mmlu(dev_result)


if __name__ == '__main__':
    main()
