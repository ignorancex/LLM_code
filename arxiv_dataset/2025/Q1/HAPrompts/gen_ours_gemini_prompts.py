"""
this is using google's sync api for gemini, it is relatively slow for large datasets like SUN-324 or ImageNet-1k
requires python >= 3.10
"""

import os
import json
import glob
import time
import argparse
import numpy as np
import google.generativeai as genai

from tqdm import tqdm
from trees.tree_utils import load_tree
from utils.directory import load_api_key
from lang_prompts.our_prompts import find_peer_nodes, find_node_to_compare, get_ancestors
from lang_prompts.our_prompts import gen_comparative_lang_prompts, gen_path_lang_prompts


def gen_1v1_comp_request(tree, prompt_generator):

    peers = dict()
    find_peer_nodes(tree, peers)
    to_compare = find_node_to_compare(peers, tree)

    batch = {k:[] for k in to_compare.keys()}
    for k, v in to_compare.items():
        query_class = k
        for related_class in v:
            prompt = prompt_generator(
                query_class, 
                related_class,
            )
            batch[query_class].append(prompt)
    request_cnt = 0
    for k, v in batch.items():
        request_cnt += len(v)
    print(f'total number of request: {request_cnt}')
    return batch


def gen_path_prompts_request(tree, prompt_generator):
    ancestors_dict = get_ancestors(tree)
    
    batch = {k:[] for k in tree.leaves()}
    for query_class, ancestors in ancestors_dict.items():
        for ancestor_class in ancestors:
            prompts = prompt_generator(
                query_class, 
                ancestor_class,
            )
            for pmt in prompts:
                batch[query_class].append(pmt)
    request_cnt = 0
    for _, v in batch.items():
        request_cnt += len(v)
    print(f'total number of request: {request_cnt}')
    return batch 


def split_batch(batch, split_id, n_split):
    batch_size = len(batch)
    mini_batch_size = int(np.ceil(batch_size / n_split))
    min_batch = batch[split_id * mini_batch_size : (split_id + 1) * mini_batch_size]
    return min_batch


def get_mini_batch(split_id, n_split, batch):
    keys = list(batch.keys())
    keys = sorted(keys)
    mini_batch_keys = split_batch(keys, split_id, n_split)
    mini_batch = {}
    for k in mini_batch_keys:
        mini_batch[k] = batch[k]
    return mini_batch


def combine_mini_batches(dataset, method, temp, max_tokens):
    method = f'{method}-{temp:.2f}-temp-{max_tokens}-mtokens'
    
    batch_request_dir = f'./image_prompts/gemini_batch_jobs/{dataset}/{method}'
    
    img_prompt = {}
    pattern = os.path.join(batch_request_dir, '*batch-download.json')
    
    for pth in glob.glob(pattern):
        with open(pth, 'r') as f:
            pts = json.load(f)
        print(f'loading {pth}')
        img_prompt.update(pts)
    return img_prompt


def generate_response_syc_mini_batch(method, mini_batch, max_tokens, temp):
    model_name = 'gemini-1.5-flash'
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temp
        )
    )
    
    stop = True if 'comp' in method else False
    img_prompts = {k:[] for k in mini_batch.keys()}
    for k, prompts in tqdm(mini_batch.items(), total=len(mini_batch)):
        for pt in prompts:
            while True:
                try:
                    if stop:
                        response = model.generate_content(
                            pt,
                            generation_config=genai.GenerationConfig(
                                stop_sequences=['.']
                            )
                        )
                    else:
                        response = model.generate_content(pt)
                    # extract text has to be in the loop
                    # gemini applies content moderation at response.text
                    # if runtime errors ocur here, try again
                    response_text =response.text
                    break
                except:
                    time.sleep(3)               
            
            img_prompts[k].append(response_text)
            
    return img_prompts




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, choices=[
        'ours-comp-gemini', 'ours-path-gemini'
    ])
    parser.add_argument('--n-split', type=int, default=10)
    opts = parser.parse_args()

    api_key = load_api_key('./data_paths.yml', 'google')
    genai.configure(api_key=api_key)

    dataset = 'imagenet'
    tree = load_tree(dataset, './trees')
    method = opts.prompt
    temp = 1.0
    max_tokens = 150
    save_mini_dir = f'./image_prompts/gemini_batch_jobs/{dataset}/{method}-{temp:.2f}-temp-{max_tokens}-mtokens/'
    if not os.path.exists(save_mini_dir): os.makedirs(save_mini_dir)

    # split original request into mini-batches
    if opts.prompt == 'ours-comp-gemini':
        full_batch = gen_1v1_comp_request(tree, gen_comparative_lang_prompts)
    else: # 'ours-path-gemini'
        full_batch = gen_path_prompts_request(tree, gen_path_lang_prompts)
    
    n_split = opts.n_split
    for start_id in range(n_split):
        img_prompts_mini_path = os.path.join(save_mini_dir, f'{start_id+1}-of-{n_split}-batch-download.json')
        mbatch = get_mini_batch(start_id, n_split, full_batch)
        
        img_prompts = generate_response_syc_mini_batch(method, mbatch, max_tokens, temp)
        with open(img_prompts_mini_path, 'w') as f:
            json.dump(img_prompts, f, indent=4)
        print(f'saving image prompts at {img_prompts_mini_path}')
    
    # combine mini-batch
    img_prompts = combine_mini_batches(dataset, method, temp, max_tokens)
    save_dir = './image_prompts/Ours/gemini/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    img_prompts_path = os.path.join(
        save_dir,
        f'{dataset}-{method}-{temp:.2f}-temp-{max_tokens}-mtokens.json'
    )

    with open(img_prompts_path, 'w') as f:
        json.dump(img_prompts, f, indent=4)
    print(f'save at {img_prompts_path}')
    

    
    