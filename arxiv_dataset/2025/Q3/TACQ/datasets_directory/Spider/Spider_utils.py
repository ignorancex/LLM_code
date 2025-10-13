import gzip
import json
import os
import os.path as osp
import ssl
import urllib.request
import re
import random


DATA_DIRECTORY = "datasets_directory/Spider/data/spider"


def format_prompt(question, db_info):
    """Create chat format with system message and schema using SimpleDDL-MD-Chat template"""
    system_msg = """### Answer the question by sqlite SQL query only and with no explanation"""
    
    # Format schema information in DDL-like comments
    schema_lines = ["### Sqlite SQL tables, with their properties:"]
    for table_idx, table_name in enumerate(db_info['table_names_original']):
        # Get columns for this table
        columns = [
            col[1] for col in db_info['column_names_original']
            if col[0] == table_idx
        ]
        schema_lines.append(f"# {table_name}({','.join(columns)});")
    
    user_content = "\n".join([
        *schema_lines,
        "###",
        f"### {question}",
        "### SQL:"
    ])
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]

def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

from torch.utils.data import Dataset as torchDataset
class Spider_N_Shot_Dataset(torchDataset):
    def __init__(self, model_name, tokenizer, data_filepath = DATA_DIRECTORY, use_train_split=False, \
                 verbose=False, \
                 make_data_wrong=False, device="cuda"):
        import pandas as pd
        import os
        self.texts = []
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = device
        # Load data files
        with open(os.path.join(data_filepath, "train_spider.json")) as f:
            train_data = json.load(f)
        with open(os.path.join(data_filepath, "dev.json")) as f:
            test_data = json.load(f)
        with open(os.path.join(data_filepath, "tables.json")) as f:
            all_tables = json.load(f)

        # Create database ID to schema mapping
        db_schema = {db['db_id']: db for db in all_tables}
        train_data_text = []
        for idx, item in enumerate(train_data):
            # Get database schema information
            db_id = item['db_id']
            db_info = db_schema[db_id]
            
            # Format input text
            input_text = format_prompt(item['question'], db_info)
            input_text = tokenizer.apply_chat_template(
                input_text,
                add_generation_prompt=True,
                tokenize=False
            )
            train_data_text.append(input_text)
            idx == 3 and print(f"{train_data_text=}")

        test_data_text = []
        for idx, item in enumerate(test_data):
            # Get database schema information
            db_id = item['db_id']
            db_info = db_schema[db_id]
            
            # Format input text
            input_text = format_prompt(item['question'], db_info)
            input_text = tokenizer.apply_chat_template(
                input_text,
                add_generation_prompt=True,
                tokenize=False
            )
            test_data_text.append(input_text)
            idx == 3 and print(f"{test_data_text=}")

        self.texts = train_data_text if use_train_split else test_data_text
    
    def shuffle(self, seed=42):
        import random
        random.seed(seed)
        random.shuffle(self.texts)

    def truncate_to_seqlen_n_samples(self, seqlen, n_samples):
        tokens_to_give = seqlen*n_samples
        current_token_count = 0
        for i in range(len(self.texts)):
            if current_token_count + len(self.tokenizer(self.texts[i])["input_ids"]) > tokens_to_give:
                break
            current_token_count += len(self.tokenizer(self.texts[i])["input_ids"])
        print(f"Truncated to {i} samples, current_token_count: {current_token_count}, tokens_to_give: {tokens_to_give}")
        self.texts = self.texts[:i]

    def to_hf_dataset(self):
        from datasets import Dataset as hfDataset
        dataset = hfDataset.from_dict({
            "text": self.texts,
        })
        return dataset
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        prompt = self.texts[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="longest", truncation=False).to(self.device)
        inputs = {k:v.squeeze(0) for k, v in inputs.items()}
        idx <= 3 and self.verbose and print(f"DATALOADER INPUT IDS SHAPE {inputs["input_ids"].shape}") # DATALOADER INPUT IDS SHAPE torch.Size([2048])
        idx <= 3 and self.verbose and print(f"INPUT IDS {inputs["input_ids"]}")
        return inputs  # a dict with input_ids, labels, attention masks, etc

def get_Spider(nsamples, seed, seqlen, model, data_filepath = DATA_DIRECTORY):
    """Follows the GPTQ repo convention for preparing a calibration dataset for Spider"""
    from transformers import AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    calibration_dataset_no_padding = Spider_N_Shot_Dataset(data_filepath=data_filepath, model_name=model, tokenizer=tokenizer, \
                use_train_split=True, make_data_wrong=False
                )
    calibration_dataset_no_padding.shuffle(seed=seed) # Same shuffle function as used in measure_importance.py
    calibration_dataset_no_padding.truncate_to_seqlen_n_samples(seqlen, nsamples)
    calibration_dataset_no_padding = calibration_dataset_no_padding.to_hf_dataset()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def custom_tokenize_function_no_padding(examples):
        return tokenizer(examples["text"], truncation=False, padding=False, max_length=seqlen, return_tensors="pt")
    calibration_dataset_no_padding = calibration_dataset_no_padding.map(custom_tokenize_function_no_padding, batched=False, num_proc=1, remove_columns=["text"])

    trainloader = []
    tokens_to_give = seqlen*nsamples
    current_token_count = 0
    for i in range(len(calibration_dataset_no_padding)):
        current_token_count += len(calibration_dataset_no_padding[i]["input_ids"][0])
        inp = calibration_dataset_no_padding[i]["input_ids"]
        inp = torch.tensor(inp)  
        tar = inp.clone()  
        tar[:-1] = -100
        i == 0 and print(f"Calibration inp {inp}")
        trainloader.append((inp, tar))
    print(f"Examples used: {len(trainloader)}; current_token_count: {current_token_count}; tokens_to_give: {tokens_to_give}")
    return trainloader, None

def get_Spider_concat(nsamples, seed, seqlen, model, data_filepath="data/spider/"):
    """
    Instead of creating a separate example for each row,
    this version concatenates texts until it hits ~seqlen tokens.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    calibration_dataset = Spider_N_Shot_Dataset(
        data_filepath=data_filepath,
        model_name=model,
        tokenizer=tokenizer,
        use_train_split=True,
        make_data_wrong=False
    )
    calibration_dataset.shuffle(seed=seed)

    calibration_dataset = calibration_dataset.to_hf_dataset()

    all_texts = calibration_dataset["text"]

    chunked_texts = []
    current_chunk = ""
    for idx, text in enumerate(all_texts):
        candidate = (current_chunk + "\n\n" + text) if current_chunk else text

        token_ids = tokenizer.encode(candidate, add_special_tokens=False)

        if len(token_ids) > seqlen:
            if current_chunk: 
                chunked_texts.append(current_chunk)
                idx < 30 and print(current_chunk)
            current_chunk = text
        else:
            current_chunk = candidate

    if current_chunk:
        chunked_texts.append(current_chunk)

    trainloader = []
    current_token_count = 0
    for chunk in chunked_texts:
        tokenized = tokenizer(
            chunk,
            truncation=True,
            max_length=seqlen,
            return_tensors="pt"
        )
        inp = tokenized["input_ids"].squeeze(0)
        tar = inp.clone()
        tar[:-1] = -100

        trainloader.append((inp, tar))
        current_token_count += len(inp)
        if len(trainloader) >= nsamples:
            break

    print(f"Built {len(trainloader)} chunks. Token count so far: {current_token_count} tokens.")
    return trainloader, None
