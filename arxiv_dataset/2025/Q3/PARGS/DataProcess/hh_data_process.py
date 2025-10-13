import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, GPT2Tokenizer, AutoModelForCausalLM
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import os
import torch.utils.data as data_utils
from transformers import GPT2Tokenizer, DataCollatorWithPadding, DefaultDataCollator, DataCollatorForSeq2Seq, RobertaTokenizer
from datasets import load_dataset, load_from_disk, Dataset # huggingface datasets
from typing import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys

"""
python hh_data_process.py
"""

def print_data(dataset, size=10):
    for i, data in enumerate(dataset):
        if i >= size:
            break
        print(data)
        
        
def split_string_at_substring(input_string):
    substring = "\n\nAssistant: "
    index = input_string.rfind(substring)
    
    if index != -1:
        first_part = input_string[:index + len(substring)]
        second_part = input_string[index + len(substring):]
        return first_part, second_part
    else:
        return "Substring not found in the input string"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("argsearch/llama-7b-rm-float32")
    tokenizer.pad_token = tokenizer.eos_token

    path = f"./"
    if not os.path.exists(path):
        os.makedirs(path)
        
    raw_datasets = load_dataset('Anthropic/hh-rlhf')
    raw_train_dataset = raw_datasets['train']
    raw_eval_dataset = raw_datasets['test']
    
    transformed_rows_train = []
    for row in raw_train_dataset:
        chosen = row['chosen']
        prompt, chosen = split_string_at_substring(chosen)
        transformed_rows_train.append({
            'prompt': prompt,
            'label': chosen,
        })
        
    transformed_rows_eval = []
    for row in raw_eval_dataset:
        chosen = row['chosen']
        prompt, chosen = split_string_at_substring(chosen)
        transformed_rows_eval.append({
            'prompt': prompt,
            'label': chosen,
        })
    
    raw_datasets_train = Dataset.from_list(transformed_rows_train)
    raw_datasets_eval = Dataset.from_list(transformed_rows_eval)
    # print_data(raw_datasets_train, size=10)
    
    # Cache
    raw_datasets_train.save_to_disk(f'{path}/hh_SFT_train')
    raw_datasets_eval.save_to_disk(f'{path}/hh_SFT_eval')









