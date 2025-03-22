from re import T
from tkinter import Label
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import csv

class IMDBDataset(Dataset):
    def __init__(self, tokenizer, raw_data, max_length):

        self.data = []
        self.labels = []
        N = len(raw_data['text'])
        for id in tqdm(range(N)):
            tmp_src = tokenizer(raw_data['text'][id], max_length = max_length, padding = 'max_length', truncation = True, return_tensors='pt')
            tmp_src['input_ids'] = tmp_src['input_ids'][0]
            tmp_src['attention_mask'] = tmp_src['attention_mask'][0]
            self.data.append(tmp_src)
            self.labels.append(raw_data['label'][id])
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item],self.labels[item]

# class IMDB_GPT(Dataset):
#     def __init__(self, tokenizer, raw_data, max_length, prompt = False, train = False):

#         self.data = []
#         self.labels = []
#         N = len(raw_data['text'])
#         self.gensrc = []
#         if prompt:
#             for id in tqdm(range(N)):
#                 text = 'Identify whether the review is positive or negative: \n' + raw_data['text'][id]
#                 if raw_data['label'] == 1:
#                     if train:
#                         text = text + '\n answer: positive'
#                     else:
#                         text = text + '\n answer:'
#                 else:
#                     if train:
#                         text = text + '\n answer: negative'
#                     else:
#                         text = text + '\n answer:'
#                 tmp_data = tokenizer.encode(text, max_length = max_length, truncation = True, padding = 'max_length')
#                 self.data.append(tmp_data)
#                 self.labels.append([-100 if x == tokenizer.pad_token_id else x for x in tmp_data])
#             self.data = torch.tensor(self.data)
#             self.labels = torch.tensor(self.labels)
#         else:
#             for id in tqdm(range(N)):
#                 tmp_data = tokenizer.encode(raw_data['text'][id], max_length = max_length, truncation = True, padding = 'max_length')
#                 self.data.append(tmp_data)
#                 self.labels.append([-100 if x == tokenizer.pad_token_id else x for x in tmp_data])
#             self.data = torch.tensor(self.data)
#             self.labels = torch.tensor(self.labels)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         return self.data[item], self.labels[item]

class IMDB_BART(Dataset):
    def __init__(self, tokenizer, raw_data, max_length):
        self.data = []
        N = len(raw_data['text'])
        for id in tqdm(range(N)):
            self.data.append(tokenizer(raw_data['text'][id], max_length = max_length, padding = 'max_length', truncation = True, return_tensors='pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def get_10class_data():
    filepath = './10class_data'
    all_data = dict()
    for split in ['train','dev','test']:
        all_data[split] = dict()
        all_data[split]['text'] = list()
        all_data[split]['label'] = list()
        with open('{}/{}.tsv'.format(filepath, split)) as f:
            for cols in f.readlines():
                cols = cols.split('\t')
                label = cols[0]
                text = cols[1]
                for id,item in enumerate(label):
                    if item == '1':
                        all_data[split]['label'].append(id)
                all_data[split]['text'].append(text)
    return all_data['train'], all_data['dev'], all_data['test']




def get_aug_data(filename, train_data, data_id = None, separate = False, label_pool = False):
    f = open('{}.txt'.format(filename),'r')
    aug_data = f.read()
    aug_data = aug_data.split('[EOSUMM]')
    if separate:
        aug_data_dict = dict()
        aug_data_dict['text'] = list()
        aug_data_dict['label'] = list()
    if data_id is not None:
        for id_train,id_raw in enumerate(data_id):
            label = int(train_data['label'][id_train]>4) if label_pool else train_data['label'][id_train] 
            if separate:
                aug_data_dict['text'].append(aug_data[id_raw])
                aug_data_dict['label'].append(label)
            else:
                train_data['text'].append(aug_data[id_raw])
                train_data['label'].append(label)
            
    else:
        for id,item in enumerate(aug_data):
            label = int(train_data['label'][id]>4) if label_pool else train_data['label'][id] 
            if separate:
                aug_data_dict['text'].append(item)
                aug_data_dict['label'].append(label)
            else:
                train_data['text'].append(item)
                train_data['label'].append(label)
    if separate:
        return aug_data_dict
    else:
        return train_data

def sample(data, data_id):
    sampled_data = dict()
    sampled_data['text'] = list()
    sampled_data['label'] = list()
    pos_count = 0
    for id in data_id:
        if data['label'][id] == 1:
            pos_count += 1
        sampled_data['text'].append(data['text'][id])
        sampled_data['label'].append(data['label'][id])
    return pos_count/len(data_id), sampled_data
