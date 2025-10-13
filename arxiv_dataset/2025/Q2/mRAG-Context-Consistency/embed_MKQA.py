import os
cache_dir = os.getenv("TMPDIR")

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import jsonlines
import argparse
from tqdm import tqdm
import numpy as np

import cohere
YOUR_COHERE_API_KEY = "YOUR_KEY"
co = cohere.Client(YOUR_COHERE_API_KEY) # Add your cohere API key from www.cohere.com

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

langs = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'ko', 'km', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh_cn']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument("--breakpoint", type=int, default=0, help="Language")

    args = parser.parse_args()
    
    lang = args.lang

    if not os.path.exists("MKQA/"):
        os.makedirs("MKQA/")

    # Load data
    #data = load_dataset("apple/mkqa", 'mkqa', split='train', cache_dir=cache_dir)
    with open('mkqa.jsonl', 'r') as f:
        data = f.read()
        data = data.split('\n')[:-1]
        data = [json.loads(i) for i in data]
    f.close()

    # continue from breakpoint
    data = data[args.breakpoint:]
    for d_index, d in enumerate(tqdm(data)):
        if {"type": "unanswerable", "text": None} in d['answers'][lang]: continue
        if {"type": "long_answer", "text": None} in d['answers'][lang]: continue

        ins = dict()
        ins['index'] = args.breakpoint + d_index
        ins['example_id'] = d['example_id']
        ins['question'] = d['queries'][lang]
        ins['gold_answers'] = []
        for i in d['answers'][lang]:
            ins['gold_answers'].append(i['text'])
            if 'aliases' in i:
                ins['gold_answers'] += i['aliases']
 
        response = co.embed(texts=[d['queries'][lang]], model='embed-multilingual-v3.0', input_type="search_query")
        query_embedding = response.embeddings
        ins['emb'] = query_embedding[0]

        with jsonlines.open(f"MKQA/{args.lang.split('_')[0]}.jsonl", mode='a') as f:
            f.write(ins)
        
if __name__ == "__main__":
    main()
