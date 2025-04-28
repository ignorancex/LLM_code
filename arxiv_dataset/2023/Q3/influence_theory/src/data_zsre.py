"""This file support run_train_model.py, run_IF_exp.py, and run_MIS_exp.py
"""
import json
import random
import torch
import logging
from src.utils import dict_to

# This class was altered from Mitchel et. al. (2022), MEND (https://github.com/eric-mitchell/mend;https://openreview.net/pdf?id=0DcZxeWfOPt)
class Create_dataset(torch.utils.data.Dataset):
    """
        PyTorch Dataset for the zsRE data
        Data format: pytorch list of tensors
    """

    def __init__(self, data_inpt, data_trg_tok, data_trg_raw, tokenizer, config):
        self.tok = tokenizer
        self.data = []
        self.config = config
        self.data = self._tokenize(
            data_inpt, data_trg_tok, data_trg_raw)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_decoder_inputs = self.data[idx]["trg_input_ids"]
        data_labels = data_decoder_inputs.masked_fill(
            data_decoder_inputs == self.tok.pad_token_id, -100)
        data_inner = {}
        data_inner["input_ids"] = self.data[idx]["inpt_input_ids"]
        data_inner["attention_mask"] = self.data[idx]["inpt_attention_mask"]
        data_inner["decoder_input_ids"] = data_decoder_inputs
        data_inner["decoder_raw"] = self.data[idx]["trg_raw"]
        data_inner["labels"] = data_labels
        data_inner['labels_mask'] = self.data[idx]["trg_mask"]
        return dict_to(data_inner, self.config.device)

    def _tokenize(self, data_inpt, data_trg_tok, data_trg_raw):
        tok_inpt = self.tok(data_inpt, return_tensors="pt", padding=True,
                            truncation=True, max_length=40).to(self.config.device)
        dataset = []
        mask_list = []
        for i in range(len(data_trg_tok)):
            mask = [1]*len(data_trg_tok[0])
            mask_list.append(mask)
        keys = ['inpt_input_ids', 'inpt_attention_mask',
                'trg_input_ids', 'trg_raw', 'trg_mask']
        for ii, iam, ti, tr, tm in zip(tok_inpt['input_ids'], tok_inpt['attention_mask'], data_trg_tok, data_trg_raw, mask_list):
            dataset.append(dict(zip(keys, [ii, iam, ti, tr, tm])))
        return dataset


def create_dataloader(data, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=False)
    return (dataloader)


def download_data(data_dir, data_seed, sample_n="all", hyperparam_search=False):
    data = json.loads(open(data_dir).read())
    # Split in half for hyperparameters and train data
    n = len(data)//2
    if hyperparam_search:
        valid_data = data[-n:]
    else:
        valid_data = data[:n]
    # If not using all data, then need to shuffle and take number of sample_n
    if sample_n != "all":
        random.seed(data_seed)
        random.shuffle(valid_data)
        valid_data = valid_data[0:sample_n]

    data_input = []
    data_answer = []
    for i, vl in enumerate(valid_data):
        data_input.append(vl['input'])
        data_answer.append(vl['output'][0]['answer'])
    return (data_input, data_answer)


def output_tok(input, model, tokenizer, config):
    data = []
    dataloader = torch.utils.data.DataLoader(
        dataset=input, batch_size=64, shuffle=False)
    for i in dataloader:
        batch = tokenizer(i, return_tensors="pt", padding=True,
                          truncation=True, max_length=40).to(config.device)
        tok_batch = batch['input_ids']
        # could make this smaller than (40)
        for one_batch in tok_batch:
            dim_pad = 40 - len(one_batch)
            data.append(torch.cat((one_batch, torch.ones(
                dim_pad).long().to(config.device)), dim=-1))
    return (data)


def extract_data_zsre(config, model, tokenizer, base_dir):
    # Download zsRE validation data set
    logging.info("Load datasets")

    # Extract Subset of Training Set
    tr_raw_inpt, tr_raw_pred = download_data(
        base_dir+"/data/zsre_train_data.json", config.data_seed, sample_n=config.n, hyperparam_search=config.hyperparam_search)
    # tokenize prediction
    tr_tok_pred = output_tok(tr_raw_pred, model, tokenizer, config)

    # Extract Test Set
    te_raw_inpt, te_raw_pred = download_data(
        base_dir+"/data/zsre_dev_data.json", config.data_seed, sample_n=config.n_test, hyperparam_search=config.hyperparam_search)
    # tokenize prediction
    te_tok_pred = output_tok(te_raw_pred, model, tokenizer, config)

    # Process data
    logging.info("Create train/test datasets")
    # Training data
    train_data = Create_dataset(tr_raw_inpt, tr_tok_pred, tr_raw_pred,
                                tokenizer, config)

    # Testing data
    test_data = Create_dataset(te_raw_inpt, te_tok_pred, te_raw_pred,
                               tokenizer, config)
    return (train_data, test_data)
