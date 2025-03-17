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
result={}
with torch.no_grad():
    for i in a.keys():
        print(i)
        sents=a[i]['sents']
        sents=[j[0].upper()+j[1:] for j in sents]
        #sents=list(zip(sents,value))
        #sents=sorted(sents,key=lambda x:-x[1])
        #value=[j[1] for j in sents]
        #sents=[j[0] for j in sents]
        entailment_map=np.zeros((len(sents),len(sents),3),dtype=float)
        pairs=[]
        for j in range(len(sents)):
            for k in range(len(sents)):
                if k!=j:
                    pairs.append([k,j])
        batch_size=512
        counter=0
        while counter<len(pairs):
            media_pair=pairs[counter:counter+batch_size]
            pre=[sents[j[0]] for j in media_pair]
            #hyp=[media[j[0]][0].upper()+media[j[0]][1:]+'.' for j in media_pair]
            hyp=[sents[j[1]] for j in media_pair]
            sample=tokenizer(pre,hyp,truncation=True,padding=True,return_tensors='pt')
            logits=nli_model(**sample.to(device))[0]
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs.detach().cpu().numpy()
            for j in range(len(media_pair)):
                entailment_map[media_pair[j][0],media_pair[j][1]]=prob_label_is_true[j]
            #break
            counter+=batch_size
        result[i]=[entailment_map,sents]
        #break
with open(args.output,'wb') as file:
    pickle.dump(result,file)
