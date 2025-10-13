import os
cache_dir = os.getenv("TMPDIR")

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import jsonlines
import argparse
from tqdm import tqdm
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
#langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
#langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']

#langs = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'ko', 'km', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh_cn']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument("--topk", type=int, default=30, help="Number of retrieved documents")

    args = parser.parse_args()
    top_k = args.topk
    batch_size = 100000

    if not os.path.exists("data_inlang_MKQA/"):
        os.makedirs("data_inlang_MKQA/")

    # Load data
    data = load_dataset("JRQi/MKQA-emb", args.lang, split="test", cache_dir=cache_dir)
    #print("Cache directory for dataset1:", data.cache_files)
    query_embs = torch.tensor(data['emb'], dtype=torch.bfloat16).to(device)

    # Mapping abbreviation between Global-MMLU and Multilingual Wikipedia
    if args.lang == "fil":
        lang_retrieval = "tl"
    else:
        lang_retrieval = args.lang
    
    # Load passages
    passages = load_dataset(f"Cohere/wikipedia-2023-11-embed-multilingual-v3", lang_retrieval, split="train", cache_dir=cache_dir)
    #print("Cache directory for dataset2:", passages.cache_files)

    all_scores = torch.tensor([[] for _ in range(len(data))], dtype=torch.bfloat16).to(device)
    all_doc = [[] for _ in range(len(data))]

    tmp_doc = []
    tmp_emb = []
    for passage_id, passage in enumerate(tqdm(passages)):
        tmp_emb.append(passage['emb'])
        # Template: "Document [{ID}](Title: {T}): {P}\n"
        tmp_doc.append({"title": passage['title'], "text": passage['text'], "url": passage['url']})

        if ((passage_id+1) % batch_size == 0) or (passage_id+1) == len(passages):
            passage_emb = torch.tensor(tmp_emb, dtype=torch.bfloat16).to(device)
            dot_scores = torch.mm(query_embs, passage_emb.T)
            all_scores = torch.cat((all_scores, dot_scores), 1)
            all_doc = [i + tmp_doc for i in all_doc]
            all_scores, top_k_hits = torch.topk(all_scores, top_k)
            all_doc = [[all_doc[idx][j] for j in i] for idx, i in enumerate(top_k_hits)]

            tmp_doc = []
            tmp_emb = []
   
    save_data = []
    for d_index, d in enumerate(tqdm(data)):
        ins = dict()
        ins['id'] = d['index']
        ins['example_id'] = d['example_id']
        ins['question'] = d['question']
        ins['gold_answers'] = d['gold_answers']

        ins['has_gold'] = False

        ins['docs'] = []
        for doc_id, doc in enumerate(all_doc[d_index]):
            tmp_doc = dict()
            tmp_doc['title'] = doc['title']
            tmp_doc['text'] = doc['text']
            tmp_doc['url'] = doc['url']
            tmp_doc['gold_answers'] = ins['gold_answers']
            tmp_doc['has_answer_text'] = False
            tmp_doc['has_answer_title'] = False
            for ans in ins['gold_answers']:
                if ans.lower() in doc['text'].lower(): tmp_doc['has_answer_text'] = True
                if ans.lower() in doc['title'].lower(): tmp_doc['has_answer_title'] = True
            tmp_doc['score'] = all_scores[d_index][doc_id].item()
            ins['has_gold'] = ins['has_gold'] or tmp_doc['has_answer_text'] or tmp_doc['has_answer_title']
            ins['docs'].append(tmp_doc)
        save_data.append(ins)
        
        '''
        with jsonlines.open(f"data/{args.lang}_gtr_top{args.topk}.jsonl", mode='a') as f:
            f.write(ins)
        '''

    
    with open(f"data_inlang_MKQA/{args.lang}_gtr_top{args.topk}.json", 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, sort_keys=False)

if __name__ == "__main__":
    main()
