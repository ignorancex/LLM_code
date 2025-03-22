# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:17:31 2023

@author: jacki
"""

import json
import os
import numpy as np
import argparse
import pickle
argparser=argparse.ArgumentParser()
argparser.add_argument('--input_file',type=str)
argparser.add_argument('--entail_file', type=str)
argparser.add_argument('--output_file',type=str)
argparser.add_argument('--num_token',type=int)
args = argparser.parse_args()
with open(args.entail_file,'rb') as file:
    data=pickle.load(file)
with open(args.input_file,'r') as file:
    a=json.load(file)
#assert 0==1
result={}
for i in a.keys():
    pos=[]
    for j in range(len(i)):
        if i[j]=='_':
            pos.append(j)
    key=i[0:pos[-1]]
    if key not in result.keys():
        result[key]=[]
    result[key].append([a[i][0],len(data[i]['sents']),a[i][1]])
token_limit=args.num_token
output={}
for i in result.keys():
    cur_len=0
    media=sorted(result[i],key=lambda x:-x[1])
    media_output=[]
    
    for j in media:
        media_ration=' '.join(j[2])
        if cur_len+len(j[0].split())+len(media_ration.split())<token_limit:
            media_output.append(j[0]+': '+media_ration)
            cur_len+=len(j[0].split())+len(media_ration.split())
    output[i]='\n'.join(media_output)
#assert 0==1
with open(args.output_file,'w') as file:
    json.dump(output,file)
