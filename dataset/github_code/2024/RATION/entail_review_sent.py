# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:09:53 2023

@author: jacki
"""

import torch
import itertools
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import time
import argparse
argparser=argparse.ArgumentParser()
argparser.add_argument('--entail_model', type=str)
argparser.add_argument('--input', type=str)
argparser.add_argument('--output', type=str)
argparser.add_argument('--output_dim', type=int)
args = argparser.parse_args()
device=torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
nli_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
nli_model=nli_model.to(device)
nli_model.eval()
with open(args.input+'.json','r') as file:
    a=json.load(file)
with torch.no_grad():
    for i in a.keys():
        print(i)
        sents=a[i]['sent']
        pairs=sents
        batch_size=512
        counter=0
        result=[]
        while counter<len(pairs):
            media_pair=pairs[counter:counter+batch_size]
            
            sample=tokenizer(media_pair,truncation=True,padding=True,return_tensors='pt')
            logits=nli_model(**sample.to(device))[0]
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs.detach().cpu().numpy()
            for j in range(len(media_pair)):
                result.append(list(prob_label_is_true[j].astype(float)))
            counter+=batch_size
        #result[i]=[entailment_map,sents]
        pairs=a[i]['pair']
        batch_size=512
        counter=0
        clique_result=[]
        while counter<len(pairs):
            media_pair=pairs[counter:counter+batch_size]
            sample=tokenizer(media_pair,truncation=True,padding=True,return_tensors='pt')
            logits=nli_model(**sample.to(device))[0]
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs.detach().cpu().numpy()
            for j in range(len(media_pair)):
                clique_result.append(list(prob_label_is_true[j].astype(float)))
            counter+=batch_size
        a[i]['sent_sent']=result
        a[i]['pair_sent']=clique_result
torch.cuda.empty_cache()
device=torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained(args.entail_model)
nli_model = AutoModelForSequenceClassification.from_pretrained(args.entail_model)
nli_model=nli_model.to(device)
nli_model.eval()
start=time.time()
result={}
with torch.no_grad():
    for i in a.keys():
        print(i)
        pos_pair=[]
        pos_sent=[]
        neg_pair=[]
        neg_sent=[]
        media=a[i]['pair']
        media_sent=a[i]['sent']
        for j in range(len(a[i]['sent'])):
            if a[i]['sent_sent'][j][2]>a[i]['sent_sent'][j][0]:
                pos_sent.append(j)
            elif a[i]['sent_sent'][j][0]>a[i]['sent_sent'][j][2]:
                neg_sent.append(j)
        for j in range(len(a[i]['pair'])):
            if a[i]['pair_sent'][j][2]>a[i]['pair_sent'][j][0]:
                pos_pair.append(j)
            elif a[i]['pair_sent'][j][0]>a[i]['pair_sent'][j][2]:
                neg_pair.append(j)
        entailment_map=np.zeros((len(a[i]['pair']),len(a[i]['sent']),int(args.output_dim)),dtype=float)
        pairs=list(itertools.product(pos_pair,pos_sent))
        pairs=pairs+list(itertools.product(neg_pair,neg_sent))
        #print(pairs)
        batch_size=512
        counter=0
        while counter<len(pairs):
            media_pair=pairs[counter:counter+batch_size]
            #print(media_pair)
            pre=[media_sent[j[1]] for j in media_pair]
            hyp=[media[j[0]][0].upper()+media[j[0]][1:]+'.' for j in media_pair]
            #hyp=[media[j[0]] for j in media_pair]
            sample=tokenizer(pre,hyp,truncation=True,padding=True,return_tensors='pt')
            logits=nli_model(**sample.to(device))[0]
            #print(logits.size())
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs.detach().cpu().numpy()
            for j in range(len(media_pair)):
                entailment_map[media_pair[j][0],media_pair[j][1]]=prob_label_is_true[j]
            #break
            counter+=batch_size
        result[i]=entailment_map
        #break
#print(time.time()-start)
with open(args.output,'wb') as file:
    pickle.dump(result,file)
