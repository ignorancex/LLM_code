#!/usr/bin/env python

import glob
import torch

def lazy_dataset_loader(pt_file, corpus_type):
    dataset = torch.load(pt_file)
    return dataset

def load_dataset(data_path, corpus_type):
    assert corpus_type in ["train", "valid", "test"]
    paragraph_len = []
    paragraph_num = []
    max_paragraph_len = 0
    pts = sorted(glob.glob(data_path + '.' + corpus_type + '.[0-9]*.pt'))
    for pt in pts:
        data = lazy_dataset_loader(pt, corpus_type)
        for ex in data:
            for paragraph in ex['src']:
                paragraph_len.append(len(paragraph))
                if len(paragraph) > max_paragraph_len:
                    max_paragraph_len = len(paragraph)
            paragraph_num.append(len(ex['src']))
    print (sum(paragraph_len)/len(paragraph_len))
    print (sum(paragraph_num)/len(paragraph_num))
    print (max_paragraph_len)

if __name__ == '__main__':
    DATA_PATH='/scratch/xxu/multi-multi/data/multi'
    #DATA_PATH='../ranked_wiki_b40/WIKI'
    load_dataset(DATA_PATH, 'train')
