import json
import pickle
import glob
from tqdm import tqdm
import re
import os

import numpy as np
    
    

def clean_set_option(text):
    if 'set_option' in text:
        return ''
    else:
        return text

def remove_repeating_lemmas(lst):
    t = set()
    temp = []
    for l in lst:
        if l['name'] not in t:
            t.add(l['name'])
            temp.append(l)
        else:
            print(l['name'])
    return temp
        

def remove_comment(text):
    idx = text.find('--')
    if idx >= 0:
        return text[:idx]
    else:
        return text

def extract_proofs(txt, clean_comment_text=False):
    blocks = re.split(r'(?=^(?:lemma|theorem)\b)', txt, flags = re.MULTILINE)
    
    header = blocks[0]
    print(txt)
    print(blocks)
    
    lst = []
    
    for block in blocks[1:]:
        match = re.match(r'^(?:lemma|theorem)\s+(\S+)', block.strip())
        name = match.group(1) if match else None
        block = re.sub(r'\n+', '\n', block)
        
        if clean_comment_text:
            block = '\n'.join([remove_comment(l) for l in block.split('\n') if remove_comment(l).strip() != ''])
        
        marker = ':= by'
        idx = block.find(marker)
        
        formal_statement = block[:idx+len(marker)]
        formal_proof = block[idx+len(marker):]
        
        t = {
            'name' : name,
            'split' : 'train',
            'header' : header,
            'informal_prefix' : '',
            'formal_statement' : formal_statement,
            'formal_proof' : formal_proof,
            }
        lst.append(t)
    return lst
        

def get_deepseek_format_proofs(lemmas):
    save_arr = []
    for lemma in lemmas:
        t = extract_proofs(lemma, clean_comment_text=True)
        save_arr.extend(t)
    return save_arr


        