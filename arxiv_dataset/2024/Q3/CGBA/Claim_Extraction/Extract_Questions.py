#!/usr/bin/env python
# coding: utf-8

import os
gpu_avail = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_avail}"


from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
path = '../COVID_pub'
name = 'COVID'


def format_inputs_QG(context: str, answer: str):
    return f"{answer} \\n {context}"

QG_model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base').cuda()
QG_model.eval()
QG_tokenizer = AutoTokenizer.from_pretrained('Salesforce/mixqg-base')


for split in ['train']:
    print(f"Extracting Questions for {split}...")
    with open(os.path.join(path,f'{name}_{split}_NEs.json'), 'r') as file:
        NER_results = json.load(file)

    
    inputs_QG = []
    sentences = []
    questions = []
    
    for text, NEs in tqdm(NER_results.items()):
        for ans in NEs:
            inputs_QG.append(format_inputs_QG(text, ans[0]))
            sentences.append(text)

    batch_size = 32
    for idx in tqdm(range(0, len(inputs_QG), batch_size)):
        texts = inputs_QG[idx:idx+batch_size]
        input_ids = QG_tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=128).input_ids.cuda()

        with torch.no_grad():  
            generated_ids = QG_model.generate(input_ids, max_length=32, num_beams=4).cpu()
            questions += QG_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        del input_ids
        del generated_ids
        torch.cuda.empty_cache()


    with open(os.path.join(path, f'{name}_{split}_Questions.json'), 'w') as file:
        json.dump(questions, file)

    with open(os.path.join(path, f'{name}_{split}_Sentences.json'), 'w') as file:
        json.dump(sentences, file)
