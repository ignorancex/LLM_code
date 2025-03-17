# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:02:58 2023

@author: jacki
"""

import torch
import itertools
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import argparse
argparser=argparse.ArgumentParser()
argparser.add_argument('--model',type=str)
argparser.add_argument('--input_file', type=str)
argparser.add_argument('--output', type=str)
args = argparser.parse_args()
print(args.model)
print(args.input_file)
print(args.output)
device=torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained(args.model)
nli_model = AutoModelForSequenceClassification.from_pretrained(args.model)
nli_model=nli_model.to(device)
nli_model.eval()
with open(args.input_file,'rb') as file:
    a=pickle.load(file)
batch_size=128
with torch.no_grad():
    for i in a.keys():
        print(i)
        sents=a[i]['sents']
        sents=[j[0].upper()+j[1:] for j in sents]
        pairs=sents
        counter=0
        media=[]
        while counter<len(pairs):
            media_pair=pairs[counter:counter+batch_size]
            
            sample=tokenizer(media_pair,truncation=True,padding=True,return_tensors='pt')
            logits=nli_model(**sample.to(device))[0]
            #print(logits.size())
            probs = logits.softmax(dim=1)[:,1]
            prob_label_is_true = list(probs.detach().cpu().numpy())
            #print(prob_label_is_true)
            #break
            #assert 0==1
            media.extend(prob_label_is_true)
            counter+=batch_size
        #result[i]=[entailment_map,sents]
        #break
        a[i]['speci']=media
with open(args.output,'wb') as file:
    pickle.dump(a,file)
