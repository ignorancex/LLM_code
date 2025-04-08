#!/usr/bin/env python
# coding: utf-8

import os
gpu_avail = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_avail}"


from tqdm import tqdm
import json
import stanza
from datasets import load_dataset, Dataset, load_from_disk
path = '../COVID_pub'
name = 'COVID'


dataset = load_from_disk(path)


stanza_nlp = stanza.Pipeline('en', use_gpu=True, processors='tokenize,ner')


not_included = ['TIME', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT']


for split in ['train']:
    print(f"Extracting Named Entities for {split}...")
    NER_results = dict()
    for text in tqdm(dataset[split]['sentences'][:]):
        pass_doc = stanza_nlp(text)
        NER_results[text] = [(ent.text, ent.type) for ent in pass_doc.ents if ent.type not in not_included]
    
    with open(os.path.join(path,f'{name}_{split}_NEs.json'), 'w') as file:
        json.dump(NER_results, file)