#!/usr/bin/env python
# coding: utf-8

import os
gpu_avail = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_avail}"


from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict
import torch
path = '../COVID_pub'
name = 'COVID'

def format_inputs_CG(question: str, answer: str):
    return f"{answer} \\n {question}"


CG_model = AutoModelForSeq2SeqLM.from_pretrained('khhuang/zerofec-qa2claim-t5-base').cuda()
CG_tokenizer = AutoTokenizer.from_pretrained('khhuang/zerofec-qa2claim-t5-base')


for split in ['train']:
    print(f"Extracting Claims for {split}...")
    
    with open(os.path.join(path,f'{name}_{split}_Questions.json'), 'r') as file:
        Question_results = json.load(file)
    with open(os.path.join(path,f'{name}_{split}_NEs.json'), 'r') as file:
        NER_results = json.load(file)
    
    NEs = [item[0] for sublist in NER_results.values() for item in sublist]
    
    inputs_CG = []
    claims = []    
        
    for question, NE in tqdm(zip(Question_results, NEs)):
            inputs_CG.append(format_inputs_CG(question, NE))

    batch_size = 64
    for idx in tqdm(range(0, len(inputs_CG), batch_size)):
        texts = inputs_CG[idx:idx+batch_size]
        input_ids = CG_tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=128).input_ids.cuda()

        with torch.no_grad():  
            generated_ids = CG_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True).cpu()
            claims += CG_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        del input_ids
        del generated_ids
        torch.cuda.empty_cache() 

    with open(os.path.join(path, f'{name}_{split}_Sentences.json'), 'r') as file:
        sentences = json.load(file)

    results = defaultdict(list)
    for sen, claim in tqdm(zip(sentences, claims)):
        results[sen].append(claim)
        
    with open(os.path.join(path, f'{name}_{split}_Claims.json'), 'w') as file:
        json.dump(results, file)

