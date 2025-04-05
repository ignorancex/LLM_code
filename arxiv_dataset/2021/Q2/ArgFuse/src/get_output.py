with open('../data/english_doc_level.txt', 'r') as f:
    x=f.readlines()

import random
random.seed(30)
import numpy as np
np.random.seed(30)

import string
def remove_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation + '£'))

true = {}
for line in x:
    if 'DOC_ID:' in line:
        key = line.split('DOC_ID: ')[-1].strip('\n')
        true[key] = {}
    if 'EVENT' in line:
        true[key]['EVENT'] = line.split(': ')[-1].strip('\n')
    if 'TIME-ARG merged:' in line:
        true[key]['TIME-ARG'] = line.split('TIME-ARG merged:')[-1].strip('\n')
    if 'PLACE-ARG merged:' in line:
        true[key]['PLACE-ARG'] = line.split('PLACE-ARG merged:')[-1].strip('\n')
    if 'CASUALTIES-ARG merged:' in line:
        true[key]['CASUALTIES-ARG'] = line.split('CASUALTIES-ARG merged:')[-1].strip('\n')
    if 'AFTER_EFFECTS-ARG merged:' in line:
        true[key]['AFTER_EFFECTS-ARG'] = line.split('AFTER_EFFECTS-ARG merged:')[-1].strip('\n')
    if 'REASON-ARG merged:' in line :
        true[key]['REASON-ARG'] = line.split('REASON-ARG merged:')[-1].strip('\n')
    if 'PARTICIPANT-ARG merged:' in line:
        true[key]['PARTICIPANT-ARG'] = line.split('PARTICIPANT-ARG merged:')[-1].strip('\n')

import pickle as pkl

with open('../resources/aggr_data-pred.pickle', 'rb') as pkl_in:
    pred = pkl.load(pkl_in)

print("--------generating reference doc------------")


arg_list = ['TIME-ARG', 'PLACE-ARG', 'CASUALTIES-ARG', 'AFTER_EFFECTS-ARG', 'REASON-ARG', 'PARTICIPANT-ARG']
#if doc_ids have no info, add N.A. (2 cases in true data)
for k, v in pred.items():
    try:
        x = true[str(k)]
        continue
    except:
        true[k] = {}
        for arg in arg_list:
            true[k][arg] = 'N.A.'
assert len(true) == len(pred)

true_sents = {}

for k, v in true.items() :
    txt = ''
    for arg in arg_list:
        if arg not in list(v.keys()): continue
        if v[arg] == 'N.A.': continue
        if txt: txt = txt + ', ' + remove_punct(v[arg].replace('N.A.', ' '))
        else: txt = remove_punct(v[arg].replace('N.A.', ' '))
        txt = ' '.join(txt.split()) #to remove extra spaces
        
    if txt: true_sents[k] = txt

#print(true_sents['525.0'])

print("--------generating hypothesis doc------------")

with open('../resources/merged_new_args.pkl', 'rb') as pkl_in:
    merged = pkl.load(pkl_in)

pred_sents = {}

for k, v in merged.items():
    txt = ''
    for arg in arg_list:
        if arg in list(v.keys()): 
            if txt : txt = txt + ', ' + ', '.join(v[arg])
            else: txt = ', '.join(v[arg])

    #print('............', txt)
        
    if txt: pred_sents[str(k)] = txt

#print(merged)
#print('============================================')
#print(pred_sents['525.0'])

print("--------writing reference & hypothesis to txt files------------")
keys = list(true_sents.keys())
ref = []
hyp = []

with open('../results/ref.txt', 'w') as f:
    for k in keys:
        ref.append([true_sents[k]])
        f.write('%s\n' %true_sents[k])

with open('../results/hyp.txt', 'w') as f:
    for k in keys:
        try:
            f.write('%s\n' %pred_sents[k])
            hyp.append(pred_sents[k])
        except Exception as e:
            f.write('%s\n' %'N.A.') #for docs not in pred
            hyp.append('\n')


print(len(ref), len(hyp))