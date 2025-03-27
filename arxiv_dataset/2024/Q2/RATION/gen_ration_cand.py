# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 23:50:02 2023

@author: jacki
"""

import json
import pickle
import numpy as np
import os
import random
import scipy.stats
import itertools
#from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import gensim
import networkx as nx
import argparse
import tqdm
#assert 0==1
stopwords = stopwords.words('english')
model=spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
random.seed(0)
argparser=argparse.ArgumentParser()
argparser.add_argument('--entail_map',type=str)
argparser.add_argument('--data',type=str)
argparser.add_argument('--entail_idx',type=int,default=1)
argparser.add_argument('--entail_thres',type=float)
argparser.add_argument('--self_sim_thres',type=float)
argparser.add_argument('--output',type=str)
#entail_thres=0.59
#self_sim_thres=0.47
#entail_idx=1
args = argparser.parse_args()
with open(args.data,'r') as file:
    a=json.load(file)

def checknumber(text):
    for i in text.strip():
        if i.isalpha():
            return True
    return False
for i in tqdm.tqdm(a.keys()):
    media=[j.lower() for j in a[i]['sent']]
    docs=model.pipe(media,disable=['parser','ner',"attribute_ruler", "lemmatizer"],batch_size=2000)
    media_sents=[]
    #print(i)
    for j in docs:
        media_sent=[]
        for k in j:
            if str(k) not in stopwords and checknumber(str(k)):
                if k.tag_.startswith('NN'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'n'))
                elif k.tag_.startswith('VB'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'v'))
                elif k.tag_.startswith('JJ'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'a'))
                elif k.tag_.startswith('RB'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'r'))
                else:
                    media_sent.append(str(k))
        #media_sent=' '.join(media_sent)  
        media_sents.append(media_sent)
    a[i]['topic_sent']=media_sents
    #texts_dict+=media_sents
for i in tqdm.tqdm(a.keys()):
    media=[j.lower() for j in a[i]['pair']]
    docs=model.pipe(media,disable=['parser','ner',"attribute_ruler", "lemmatizer"],batch_size=2000)
    media_sents=[]
    #print(i)
    for j in docs:
        media_sent=[]
        for k in j:
            if str(k) not in stopwords and checknumber(str(k)):
                if k.tag_.startswith('NN'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'n'))
                elif k.tag_.startswith('VB'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'v'))
                elif k.tag_.startswith('JJ'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'a'))
                elif k.tag_.startswith('RB'):
                    media_sent.append(lemmatizer.lemmatize(str(k),'r'))
                else:
                    media_sent.append(str(k))
        #media_sent=' '.join(media_sent)  
        media_sents.append(list(set(media_sent)))
    a[i]['topic_pair']=media_sents
with open(args.entail_map,'rb') as file:
    b=pickle.load(file)
for i in a.keys():
    a[i]['rev_map']=b[i]
del b

result={}
def gen_clique(map1,entail_thres,self_sim_thres,entail_idx):
    modify_map=np.zeros((map1.shape[0],map1.shape[1]))
    for j in range(map1.shape[0]):
        for k in range(map1.shape[1]):
            if map1[j,k,entail_idx]>entail_thres:
                if map1[j,k,entail_idx]>modify_map[j,k]:
                        modify_map[j,k]=1.0
    m=modify_map.T
    d = m.T @ m
    norm = (m * m).sum(0, keepdims=True) ** .5+1e-3
    cor_map=d / norm / norm.T
    
    graph=nx.Graph()
    for j in range(cor_map.shape[1]):
        graph.add_node(j)
    for j in range(cor_map.shape[1]):
        for k in range(j,cor_map.shape[1]):
            if cor_map[j,k]>self_sim_thres and j!=k:
                graph.add_edge(j,k)
    clique_list=list(nx.connected_components(graph))
    clique_list=[list(j) for j in clique_list]
    repre_list=select(clique_list,modify_map)
    opin2clique={}
    for j in range(len(clique_list)):
       for k in clique_list[j]:
            opin2clique[k]=j
    assert len(opin2clique.keys())==map1.shape[0]
    clique_map=np.zeros((len(clique_list),map1.shape[1]))
    for j in range(map1.shape[0]):
        for k in range(map1.shape[1]):
            if map1[j,k,entail_idx]>entail_thres:
                if map1[j,k,entail_idx]>clique_map[opin2clique[j],k]:
                        clique_map[opin2clique[j],k]=map1[j,k,entail_idx]
    return clique_list,repre_list,clique_map
    
def select(clique_list,map1):
    repre_list=[]
    for j in range(len(clique_list)):
        if len(clique_list[j])==1:
            repre_list.append(clique_list[j][0])
        else:
            media_map=np.sum(map1[clique_list[j]]>0.33,axis=1)
            #print(media_map.shape)
            idx=np.argmax(media_map)
            repre_list.append(clique_list[j][idx])
    return repre_list
cov=0.0
result={}
min_pairs=[]
keywords={i:[] for i in a.keys()}
npmi=[]
for i in a.keys():
    result[i]=[]
    #print(i)
    map1=a[i]['rev_map']
    for j in range(map1.shape[0]):
        for k in range(map1.shape[1]):
            if map1[j,k,args.entail_idx]>1.0:
                map1[j,k,args.entail_idx]=1.0
    clique_list,repre_list,clique_map=gen_clique(map1,args.entail_thres,args.self_sim_thres,args.entail_idx)
    filter_counter={}
    sent_id_list={}
    for j in range(len(clique_list)):
        sent_id_list[j]=[]
        filter_counter[j]=0
    media_map=np.argmax(clique_map,axis=0)
    for j in range(media_map.shape[0]):
        if clique_map[media_map[j],j]>0:
            sent_id_list[media_map[j]].append(j)
            filter_counter[media_map[j]]+=1
    min_len=0
    while min_len<5 or len(sent_id_list)>9:
        min_len=9999
        min_id=-1
        for j in sent_id_list.keys():
            if filter_counter[j]<min_len:
                min_id=j
                min_len=filter_counter[j]
        if min_len<5 or len(sent_id_list)>9:
            clique_map[min_id]=0.0
            for j in sent_id_list[min_id]:
                max_idx=np.argmax(clique_map[:,j])
                if clique_map[max_idx,j]>0:
                    sent_id_list[max_idx].append(j)
                    filter_counter[max_idx]+=1
            del(sent_id_list[min_id])
            del(filter_counter[min_id])
    for j in range(clique_map.shape[0]):
        if j not in sent_id_list.keys():
            clique_map[j]=0.0
    clique_map=clique_map/np.sum(clique_map,axis=0,keepdims=True)
    media_cov=0.0
    for j in range(len(clique_list)):
        if j in sent_id_list.keys():
            media_cov+=float(len(sent_id_list[j]))
            media={'topic_sents':[a[i]['topic_sent'][s] for s in sent_id_list[j]],
                   'topic_clique':[a[i]['topic_pair'][s] for s in sorted(clique_list[j])],
                   'sents':[a[i]['sent'][s] for s in sent_id_list[j]],
                   'clique':[a[i]['pair'][s] for s in sorted(clique_list[j])],
                   'clique_value':[clique_map[j,s] for s in sent_id_list[j]],
                   'value':[map1[sorted(clique_list[j]),s] for s in sent_id_list[j]],
                   'repre':a[i]['pair'][repre_list[j]]}
            result[i].append(media)
            #assert 0==1
    media_cov=media_cov/float(len(a[i]['sent']))
    cov+=media_cov
    dict_list=a[i]['topic_sent']
    gensim_dict=corpora.Dictionary(dict_list)
    gensim_dict.filter_extremes(no_below=2,no_above=0.75)
    texts_dict=[]
    for j in range(len(result[i])):
        media=[]
        for s in result[i][j]['topic_sents']:
            media+=s
        texts_dict.append(media)
    corpus = [gensim_dict.doc2bow(line) for line in texts_dict]
    model = TfidfModel(corpus)
    id2word={}
    for j in gensim_dict.token2id.keys():
        id2word[gensim_dict.token2id[j]]=j
    for j in range(len(result[i])):
        media=result[i][j]
        topic_sents=None
        for k in range(len(media['topic_sents'])):
            media_vector=media['topic_sents'][k]
            media_vector=gensim_dict.doc2bow(media_vector)
            media_vector=model[media_vector]
            media_vector=gensim.matutils.corpus2dense([media_vector], num_terms=len(gensim_dict.token2id))[:,0]
            if topic_sents is None:
                topic_sents=media_vector
            else:
                topic_sents+=media_vector
        keyword=topic_sents
        keyword1=np.argsort(-keyword)
        final_keyword=[]
        for k in keyword1[0:5]:
            if keyword[k]>0.0:
                final_keyword.append(k)
        media['keyword']=[id2word[k] for k in final_keyword]
        keywords[i].append([id2word[k] for k in final_keyword])
        clique_word=[]
        for s in media['topic_clique']:
            clique_word.extend(s)
        clique_word=set(clique_word)
        keyword1=np.argsort(-keyword)
        #counter=0
        final_keyword=[]
        for k in keyword1:
            if keyword[k]>0.0:
                if id2word[k] not in clique_word:
                    final_keyword.append(k)
                    if len(final_keyword)>=5:
                        break
            else:
                break
        keyword=[id2word[k] for k in final_keyword]
        media['keyword']=keyword
        keyword_value={keyword[k]:topic_sents[final_keyword[k]] for k in range(len(final_keyword))}
        ratio=[]
        for s in range(len(media['topic_sents'])):
            count=0
            media_vector=gensim_dict.doc2bow(media['topic_sents'][s])
            for k in media['topic_sents'][s]:
                if k in keyword:
                    count+=keyword_value[k]
            media_sum=[]
            for k in media_vector:
                media_sum.append(topic_sents[k[0]]*k[1])
            if sum(media_sum)>0.0:
                ratio.append(float(count)/float(sum(media_sum)))
            else:
                ratio.append(0.0)
        media['ratio_r']=ratio
        ratio=[]
        for s in range(len(media['topic_sents'])):
            media1=set(media['topic_sents'][s])
            media1=set(keyword).intersection(media1)
            ratio.append(sum([keyword_value[k] for k in media1])/sum([keyword_value[k] for k in keyword]))    
        media['ratio_p']=ratio
        media_topic_sents={}
        for k in range(topic_sents.shape[0]):
            if topic_sents[k]>0.0:
                media_topic_sents[id2word[k]]=topic_sents[k]
        media['topic_value']=media_topic_sents
    cm = CoherenceModel(topics=keywords[i], texts=a[i]['topic_sent'], dictionary=gensim_dict, coherence='c_npmi',processes=1)
    coherence = cm.get_coherence()
    npmi.append(coherence)
print('NPMI: '+str(np.mean(npmi)))
result1={}
for i in result.keys():
    for j in range(len(result[i])):
        #del(result[i][j]['topic_sents'])
        del(result[i][j]['topic_clique'])
        result1[i+'_'+str(j)]=result[i][j]
with open(args.output,'wb') as file:
    pickle.dump(result1,file)
