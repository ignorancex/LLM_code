import json
import time
import argparse
import os
import multiprocessing
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--domain', type=str, required=True)
    args = parser.parse_args()
    
    output_folder = f'outputs/tmp_response/'
    output_file = f'outputs/{args.model_name}_{args.domain}_selected_chapters.json'
    
    prompt_template = """You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the provided data. 
The question can be found below. Given the table of content of the book {book}, please select the chapters that you find useful in solving the question.
Please provide an explanation supporting your choice. At the last line of your response, format the number of the chapters with a list, like '[0]'. Limit the number of chapters to at most 1. Output '[]' if none of the chapters are useful. The last line should start with '[' and end with ']'.

Question:
{question}

Table of Content:
{content}

Response:
"""

    domain2book = {
        'causality': 'Introduction to Causal Inference',
        'physics': 'University Physics',
        'chemistry': "Atkins' Physical Chemistry",
    }
    book = domain2book[args.domain]
    structure = json.load(open(f'../tool_creation/outputs/structure_{args.domain}.json'))

    content = ''
    for idxi, i in enumerate(structure):
        content += str(idxi) + '. ' + i + '\n'
    
    if args.domain == 'causality':
        data = json.load(open(f'../../evaluation_data/qrdata_causal.json'))
    elif args.domain == 'physics':
        data = json.load(open(f'../../evaluation_data/theoremqa_phy.json'))
    elif args.domain == 'chemistry':
        data = json.load(open(f'../../evaluation_data/scibench_chem.json'))

    prompts = []
    for idxi, i in enumerate(data):
        if args.domain == 'causality':
            question=i['data_description'] + '\n' + i['question'] 
        elif args.domain == 'physics':
            question=i['Question']
        elif args.domain == 'chemistry':
            question=i['problem_text']
        prompt = prompt_template.format(book=book, question=question, content=content.strip()).strip()
        prompts.append(prompt)
        data[idxi]['prompt'] = prompt
    
    all_responses = run_inference(prompts, output_folder, args)
   
    for idx, i in enumerate(data):
        if idx >= len(all_responses):
            break
        data[idx]['output'] = all_responses[idx]
            
    json.dump(data, open(output_file, 'w'), indent = 4)
    
    remove_tmp_files(output_folder)
