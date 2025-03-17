# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:39:15 2023

@author: jacki
"""

import json
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from nltk.corpus import stopwords
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('pair_file')
parser.add_argument('sent_file')
parser.add_argument('output')
args=parser.parse_args()
nlp=spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
en_stop_words = stopwords.words('english')
en_stop_words=[i for i in en_stop_words if i not in ['not','no']]
#result={}
with open(args.pair_file,'r') as file:
    a=json.load(file)
#assert 0==1
pairs={}
for i in a.keys():
    pairs[i]=[]
    for j in a[i].keys():
        pairs[i].extend(a[i][j])
result1={}
#assert 0==1
for i in pairs.keys():
    result1[i]={}
    media_noun=[j[0].lower().split() for j in pairs[i]]
    media_noun=[j[:-1]+[lemmatizer.lemmatize(j[-1])] for j in media_noun]
    media_noun=[' '.join(j) for j in media_noun]
    pairs[i]=[media_noun[j]+' is '+pairs[i][j][1] for j in range(len(pairs[i]))]
    pairs[i]=[j.lower().strip() for j in pairs[i]]
    pairs[i]=list(set(pairs[i]))
    noun_list=[j.split(' is ')[0].strip() for j in pairs[i]]
    adj_list=[j.split(' is ')[1].strip() for j in pairs[i]]
    mask_list=[0]*len(noun_list)
    for j in range(len(noun_list)):
        doc=nlp(noun_list[j])
        noun_chunks=list(doc.noun_chunks)
        if len(noun_chunks)==1:
            for n in noun_chunks:
                if n.root.tag_ in ['NN','NNS','NNPS','NNP']:
                    noun_list[j]=lemmatizer.lemmatize(str(n.root))
    for j in range(len(adj_list)):
        doc=adj_list[j].split()
        media_word=[]
        #assert 0==1
        for a in doc:
            if a not in en_stop_words:
                media_word.append(str(a))
        if len(media_word)>0:
            adj_list[j]=' '.join(media_word)
    key_list=[' '.join([noun_list[j],adj_list[j]]) for j in range(len(noun_list))]
    for j in range(len(key_list)):
        for k in range(len(key_list)):
            if j!=k:
                if key_list[j]==key_list[k] and mask_list[j]!=1 and mask_list[k]!=1:
                    #print(pairs[i][j])
                    #print(pairs[i][k])
                    if len(pairs[i][j].split())>len(pairs[i][k].split()):
                        mask_list[j]=1
                    else:
                        mask_list[k]=1
    result1[i]['pairs']=[pairs[i][j] for j in range(len(pairs[i])) if mask_list[j]!=1]
    #print(len(result1[i]['pairs']))
    #print(i)
#assert 0==1

with open(args.sent_file,'r') as file:
    b=json.load(file)
for i in b:
    entity_id=i['entity_id']
    sent_list=[]
    for j in i['reviews']:
        for k in j['sentences']:
            if len(k.split())>=3 and len(k.split())<=30:
                sent_list.append(k.strip())
    result1[entity_id]['sent']=sent_list
    result1[entity_id]['pair']=result1[entity_id]['pairs']
    del(result1[entity_id]['pairs'])
with open(args.output,'w') as file:
    json.dump(result1,file)