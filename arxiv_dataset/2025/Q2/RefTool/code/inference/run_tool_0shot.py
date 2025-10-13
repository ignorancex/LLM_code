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

def get_tool_input_all(data, args):
    skills = json.load(open(f'../../generated_tools/{args.skill_model_name}_{args.domain}_tools.json', 'r'))
    selected_functions = json.load(open(f'outputs/{args.model_name}_{args.domain}_selected_tools.json', 'r'))
        
    tool_template = "Please note that we provide you several functions for the above question. If the functions are related to the question, you are encouraged to use the functions to solve the question. The functions will also be provided in execution, so just call them. *DO NOT* define the functions again or import the functions.\n\nFunctions:\n{functions}"
    
    skill_template_qrdata = """{idx}.
Function Description:
{description}

Function:
{function}

"""

    skill_template = """{idx}.
Function Description:
{description}

Function:
{function}

Example Question:
{example_question}

Example Solution:
{example_solution}

"""

    if args.domain == 'causality':
        input_prompts = [get_pot_input_qrdata(d) for d in data]
    elif args.domain == 'physics':
        input_prompts = [get_pot_input_theoremqa(d) for d in data]
    elif args.domain == 'chemistry':
        input_prompts = [get_pot_input_scibench(d) for d in data]
    
    prompts = []
    function_lists = []
    all_functions = ""
    for idxi, i in enumerate(range(len(data))):
        function_list = []
        for p in selected_functions[idxi]['pred']:
            chapter_function_list = p['function_list']
            try:
                selected_list = json.loads('['+p['output'].split('[')[1].split(']')[0]+']')
                # print(selected_list)
                for j in selected_list:
                    function_list.append(skills[chapter_function_list[j]])
            except:
                continue
        
        if len(function_list) > 0:
            functions, full_function = "", ""
            for idxs, s in enumerate(function_list):
                if args.domain == 'causality':
                    functions += skill_template_qrdata.format(idx=idxs, description=s['description'], function=s['function'])
                else:
                    functions += skill_template.format(idx=idxs, description=s['description'], function=s['function'], example_question=s['example']['question'], example_solution=s['example']['solution'])                    

            prompts.append(input_prompts[i].split('Response:')[0] + tool_template.format(functions = functions) + 'Response:\n')
            function_lists.append(function_list)
        else:
            prompts.append(input_prompts[i])
            function_lists.append([])
    
    return prompts, function_lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--skill_model_name', type=str, default='gpt4o')
    parser.add_argument('--max_tokens', type=int, default=2048)
    args = parser.parse_args()
    
    path = f'outputs/'
    if args.domain == 'causality':
        data = json.load(open(f'../../evaluation_data/qrdata_causal.json'))
    elif args.domain == 'physics':
        data = json.load(open(f'../../evaluation_data/theoremqa_phy.json'))
    elif args.domain == 'chemistry':
        data = json.load(open(f'../../evaluation_data/scibench_chem.json'))
    
    all_prompts, function_lists = get_tool_input_all(data, args)
  
    output_folder = os.path.join(path, 'tmp_response')
    all_responses = run_inference(all_prompts, output_folder, args)
    
    for idx in range(len(data)):
        data[idx]['function_list'] = function_lists[idx]
        data[idx]['output'] = all_responses[idx]
            
    json.dump(data, open(os.path.join(path, f'{args.model_name}_{args.domain}_tool_0shot.json'), 'w'), indent = 4)

    remove_tmp_files(output_folder)