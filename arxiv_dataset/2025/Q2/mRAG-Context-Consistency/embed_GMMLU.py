import os
cache_dir = os.getenv("TMPDIR")

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import jsonlines
import argparse
from tqdm import tqdm
import numpy as np

import cohere
YOUR_COHERE_API_KEY = "YOUR_KEY"
co = cohere.Client(YOUR_COHERE_API_KEY) # Add your cohere API key from www.cohere.com

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
#langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
#langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", help="Language")

    args = parser.parse_args()
    
    if not os.path.exists("GMMLU/"):
        os.makedirs("GMMLU/")

    # Load data
    data = load_dataset("CohereForAI/Global-MMLU", args.lang, cache_dir=cache_dir)['test']
    
    for d_index, d in enumerate(tqdm(data)):
        ins = dict()
        ins['id'] = d['sample_id']
        ins['subject'] = d['subject']
        ins['subject_category'] = d['subject_category']
        ins['question'] = d['question']
        ins['option_a'] = d['option_a']
        ins['option_b'] = d['option_b']
        ins['option_c'] = d['option_c']
        ins['option_d'] = d['option_d']
        ins['answer'] = d['answer']
        
        response = co.embed(texts=[d['question']], model='embed-multilingual-v3.0', input_type="search_query")
        query_embedding = response.embeddings
        ins['emb'] = query_embedding[0]

        with jsonlines.open(f"GMMLU/{args.lang}.jsonl", mode='a') as f:
            f.write(ins)
        
if __name__ == "__main__":
    main()
