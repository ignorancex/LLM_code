"""This file support run_train_model.py, run_IF_exp.py, and run_MIS_exp.py
"""
import json
from src.utils import dict_to
import torch
import logging


# This class was altered from Mitchel et. al. (2022) (https://github.com/eric-mitchell/mend;https://openreview.net/pdf?id=0DcZxeWfOPt)
class Create_dataset(torch.utils.data.Dataset):
    """
        PyTorch Dataset for the wikiText data
        Data format: pytorch list of tensors
    """

    def __init__(self, data_inpt, tokenizer, config):
        self.tok = tokenizer
        self.data = []
        self.config = config
        self.n_tokens = 10  # Generate the last 10 tokens of each paragraph
        self.data = self._tokenize(data_inpt)

    def __len__(self):
        return len(self.data)

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tok.pad_token_id).any():
            raise ValueError("Left-padding not supported for GPT2")

    def get_wiki_labels(self, ids):
        self._check_padding(ids)

        labels = ids.clone()
        end_idxs = (labels != self.tok.pad_token_id).sum(-1)
        for batch_idx, end_idx in enumerate(end_idxs):
            labels[batch_idx, :end_idx - self.n_tokens] = -100
        labels[labels == self.tok.pad_token_id] = -100
        return labels

    def __getitem__(self, idx):
        data_inner = {}
        data_inner["input_ids"] = self.data[idx]["inpt_input_ids"]
        data_inner["attention_mask"] = self.data[idx]["inpt_attention_mask"]
        data_inner["labels"] = self.data[idx]["labels"]
        return dict_to(data_inner, self.config.device)

    def _tokenize(self, data_inpt):
        tok_inpt = self.tok(data_inpt, padding=True,
                            return_tensors="pt", truncation=False, max_length=100)
        tok_inpt['labels'] = self.get_wiki_labels(tok_inpt["input_ids"])

        dataset = []
        keys = ['inpt_input_ids', 'inpt_attention_mask', 'labels']
        for ii, iam, lb in zip(tok_inpt['input_ids'], tok_inpt['attention_mask'], tok_inpt['labels']):
            dataset.append(dict(zip(keys, [ii, iam, lb])))
        return dataset


def extract_data_wiki(config, tokenizer, base_dir):
    # Extract Subset of Training Set
    with open(base_dir+'/data/wiki_train_data.json', 'r') as f:
        train_data = json.load(f)

    # Extract Subset of Test Set
    with open(base_dir+'/data/wiki_dev_data.json', 'r') as f:
        test_data = json.load(f)

    # Process data
    logging.info("Create train/test datasets")
    # Training data
    train_dataset = Create_dataset(
        train_data[0:config.n], tokenizer, config)

    # Testing Data
    test_dataset = Create_dataset(
        test_data[0:config.n_test], tokenizer, config)

    return (train_dataset, test_dataset)
