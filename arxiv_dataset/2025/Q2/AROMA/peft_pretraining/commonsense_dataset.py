import logging
import os
from typing import Dict, Sequence, Union, List
import datasets
import torch
from datasets import load_dataset
import transformers

from .base_dataset import IGNORE_INDEX

logger = logging.getLogger(__name__)

def generate_prompt(data_point):
    
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""

def build_commonsense_dataset(
    data_path: Union[List[str], str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    val_set_size: int = 0,
    preprocessing_num_workers = None,
):
    
    def tokenize(prompt, add_eos_token=True):
        
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_seq_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        
        full_prompt = generate_prompt(data_point)
        return tokenize(full_prompt)

    logger.info("Building Commonsense dataset...")
    
    
    if isinstance(data_path, (list, tuple)):
        data_path = data_path[0]
        
    
    if data_path.endswith(".json"):
        raw_dataset = load_dataset("json", data_files=data_path)
    else:
        raw_dataset = load_dataset(data_path)
    
    
    tokenized_dataset = raw_dataset.map(
        generate_and_tokenize_prompt,
        # batched=False,
        # num_proc=preprocessing_num_workers,
        remove_columns=raw_dataset["train"].column_names,
        # keep_in_memory=False,
        # desc="Preprocessing Commonsense dataset",
    )
    processed_dataset = tokenized_dataset
    
    
    if val_set_size > 0:
        train_val = processed_dataset["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        return train_val["train"], train_val["test"]
    
    return processed_dataset["train"], None
