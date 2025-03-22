# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:26:54 2023

@author: jacki
"""

from summac.model_summac import SummaCZS, SummaCConv
import json
import pickle
import numpy as np
import argparse
argparser=argparse.ArgumentParser()
argparser.add_argument('--input', type='str')
args = argparser.parse_args()
model_conv = SummaCConv(models=["vitc-only"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
with open(args.input,'rb') as file:
    a=pickle.load(file)
summs=[]
docs=[]
for i in a.keys():
    repre=a[i]['repre']+'.'
    repre=repre[0].upper()+repre[1:].lower()
    document=' '.join([j[0].upper()+j[1:] for j  in a[i]['sents']])
    summs.append(repre)
    docs.append(document)
    #break
score_conv2 = model_conv.score(docs, summs)
#print(score_conv2)
print('Summac:'+str(np.mean(score_conv2['scores'])))
