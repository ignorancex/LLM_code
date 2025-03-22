# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:31:54 2023

@author: jacki
"""

import torch
import itertools
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import pickle
from sklearn import metrics
import argparse
import tqdm
argparser=argparse.ArgumentParser()
argparser.add_argument('--input',type=str)
args = argparser.parse_args()
device=torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained(r'princeton-nlp/unsup-simcse-roberta-large')
nli_model = AutoModel.from_pretrained(r'princeton-nlp/unsup-simcse-roberta-large')
nli_model=nli_model.to(device)
nli_model.eval()
with open(args.input,'r') as file:
    media=json.load(file)
cos_sim=0.0
div_sim=0.0
print(args.input)
with torch.no_grad():
    for i in tqdm.tqdm(media.keys()):
        sample1 =tokenizer(media[i][1],truncation=True,padding=True,return_tensors='pt').to(device)
        embedding1 = nli_model(**sample1, output_hidden_states=True, return_dict=True).pooler_output
        sample2 =tokenizer([media[i][0]],truncation=True,padding=True,return_tensors='pt').to(device)
        embedding2 = nli_model(**sample2, output_hidden_states=True, return_dict=True).pooler_output
        #m=media.T
        d = embedding2 @ embedding1.T
        norm1 = (embedding1 * embedding1).sum(1, keepdims=True) ** .5+1e-3
        norm2 = (embedding2* embedding2).sum(1, keepdims=True) ** .5+1e-3
        cos_sim_media=d / norm2 / norm1.T
        for j in range(cos_sim_media.shape[1]):
            cos_sim+=cos_sim_media[0,j].item()
        if len(media[i][1])>1:
            d = embedding1 @ embedding1.T
            div_sim_media=d/norm1/norm1.T
            for j in range(div_sim_media.shape[1]):
                for k in range(j+1,div_sim_media.shape[1]):
                    div_sim+=div_sim_media[j,k].item()
print('Relatedness:'+str(cos_sim/len(media)/len(media[i][1])))
if div_sim>0.0:
    print('Diversity:'+str(1.0-div_sim/len(media)/len(media[i][1])))
    
