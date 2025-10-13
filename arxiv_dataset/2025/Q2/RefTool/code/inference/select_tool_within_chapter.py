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
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--skill_model_name', type=str, default='gpt4o')
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()
    
    output_folder = f'outputs/tmp_response/'
    output_file = f'outputs/{args.model_name}_{args.domain}_selected_tools.json'
    
    prompt_template = """You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question.  
The question and the list of skills can be found below. Please select the skills that you find useful in solving the question.
Please provide an explanation supporting your choice. At the last line of your response, format the number of the skills with a list, like '[0]'. Limit the number of skills to at most {max_skills}. Output '[]' if none of the skills are useful. The last line should start with '[' and end with ']'.

Question:
{question}

List of skills:
{skills}
Response:
"""

    skill_template = """{idx}.
Skill Description:
{description}

Function:
{function}

Example Question:
{example_question}

Example Solution:
{example_solution}

"""
    
    max_skills = 1
    if args.domain == 'causality':
        data = json.load(open(f'../../evaluation_data/qrdata_causal.json'))
    elif args.domain == 'physics':
        data = json.load(open(f'../../evaluation_data/theoremqa_phy.json'))
    elif args.domain == 'chemistry':
        data = json.load(open(f'../../evaluation_data/scibench_chem.json'))
        max_skills = 2
        
    prompts = []
    num_skills = []
    for idxi, i in enumerate(data):
        data[idxi]['pred'] = []
        selected_chapters = json.load(open(f'outputs/{args.model_name}_{args.domain}_selected_chapters.json'))

        structure = json.load(open(f'../tool_creation/outputs/structure_{args.domain}.json'))
        id2chapter = {}
        for idxs, s in enumerate(structure):
            id2chapter[idxs] = s

        skills = json.load(open(f'../../generated_tools/{args.skill_model_name}_{args.domain}_tools.json', 'r'))

        try:
            chapter_list = json.loads('['+selected_chapters[idxi]['output'].split('[')[1].split(']')[0]+']')
        except:
            continue

        for c in chapter_list:
            try:
                chapter_name = id2chapter[c]
            except:
                continue
            filtered_functions = []
            function_list = []
            for idxs, s in enumerate(skills):
                if s['chapter_name'] == chapter_name:
                    filtered_functions.append(s)
                    function_list.append(idxs)
            num_skills.append(len(function_list))
            
            cur_skills = ''
            for idxs, s in enumerate(filtered_functions):
                cur_skills += skill_template.format(idx=idxs, description=s['description'], function=s['function'], example_question=s['example']['question'], example_solution=s['example']['solution'])

            if args.domain == 'causality':
                question=i['data_description'] + '\n' + i['question'] 
            elif args.domain == 'physics':
                question=i['Question']
            elif args.domain == 'chemistry':
                question=i['problem_text']
                
            if len(cur_skills) > 0:
                prompt = prompt_template.format(max_skills=max_skills, question=question, skills=cur_skills.strip()).strip()
                prompts.append(prompt)

                # one prompt each chapter (if there are candidate skills in this chapter)
                data[idxi]['pred'].append({
                    'prompt_idx': len(prompts)-1,
                    'chapter_name': chapter_name,
                    'function_list': function_list})
                    
    print('number of prompts:', len(prompts))
    print('average number of skills for selection:', np.mean(num_skills))
    print('max number of skills for selection:', np.max(num_skills)) 
    
    all_responses = run_inference(prompts, output_folder, args)
            
    for idx, i in enumerate(data):
        for idxp in range(len(data[idx]['pred'])):
            data[idx]['pred'][idxp]['output'] = all_responses[data[idx]['pred'][idxp]['prompt_idx']]
            
    json.dump(data, open(output_file, 'w'), indent = 4)
    
    remove_tmp_files(output_folder)