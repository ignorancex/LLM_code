# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:09:53 2023

@author: jacki
"""
#assert 0==1
import torch
import itertools
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import pickle
from sklearn import metrics
import argparse
argparser=argparse.ArgumentParser()
argparser.add_argument('--input',type='str')
args = argparser.parse_args()
device=torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-large')
nli_model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-large')
nli_model=nli_model.to(device)
nli_model.eval()
with open(args.input,'rb') as file:
    a=pickle.load(file)
texts={}
labels={}
for i in a.keys():
    if i.split('_')[0] not in texts.keys():
        texts[i.split('_')[0]]=[]
        labels[i.split('_')[0]]=[]
    texts[i.split('_')[0]].extend(a[i]['sents'])
    if len(labels[i.split('_')[0]])>0:
        counter=max(labels[i.split('_')[0]])+1
    else:
        counter=0
    labels[i.split('_')[0]].extend([counter for j in range(len(a[i]['sents']))])    
scores=0.0
scorech=0.0
scoredb=0.0
with torch.no_grad():
    for i in texts.keys():
        text=texts[i]
        batch_size=512
        counter=0
        media=[]
        while counter<len(text):
            media_pair=text[counter:counter+batch_size]
            sample1=tokenizer(media_pair,truncation=True,padding=True,return_tensors='pt').to(device)
            embeddings = nli_model(**sample1, output_hidden_states=True, return_dict=True).pooler_output
            #embeddings = embeddings.detach().cpu().numpy()
            media.extend(list(embeddings))
            counter+=batch_size
            #break
        media=torch.stack(media)
        media=media.detach().cpu().numpy()
        score=metrics.silhouette_score(media, labels[i],metric='cosine')
        #print(i)
        #print(score)
        scores+=score
        """
        score=metrics.calinski_harabasz_score(media, labels[i])
        scorech+=score
        score=metrics.davies_bouldin_score(media,labels[i])
        scoredb+=score
        """
scores=scores/250.0
print('Silhouette:'+str(scores))

