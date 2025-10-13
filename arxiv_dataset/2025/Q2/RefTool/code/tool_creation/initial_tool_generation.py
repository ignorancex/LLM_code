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
    parser.add_argument('--force_generate', action='store_true', help='force generate the output file')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--domain', type=str, required=True)
    args = parser.parse_args()
    
    output_folder = f'outputs/tmp_response/'  # path to save temp outputs
    
    prompt_template = """Please extract the skills from the following text. The text is a section from the chapter {chapter} of the book {book}.
Each skill is a python function with comments of parameters and returns, accompanied by a description and a demonstration example of using the skill. 
Please limit the number of skills to 2, and organize the skills in a list of json objects.
Please implement the function, and *do not* leave it as a placeholder. Note the indent in code is 4 spaces. All packages used should be imported inside the function. The function should be self-contained.
If the text contains examples, you are encouraged to use the examples in the text, otherwise please design examples by yourself. The answer to the example question is encouraged to be numerical.
NOTE THAT THE SKILL PYTHON CODE SHOULD NOT BE SPECIFIC TO/ONLY APPLIED TO THE CHOSEN EXAMPLE! PLEASE GENERATE GENERAL SKILL CODE.
The output should be in *complete* json structure, starting with '[' and ending with ']'.

Example output:
{example}

Text:
{text}
"""
    
    example = [
        {
            "description": "Compute the expected return using the Capital Asset Pricing Model (CAPM) formula.",
            "function": """def expected_return(rf, beta, rm): 
    \"\"\" 
    Parameters: 
    - rf (float): The risk-free rate. 
    - beta (float): The beta of the portfolio. 
    - rm (float): The return on the market.
    Returns: 
    - float: The expected return.
    \"\"\" 
    return rf + beta * (rm - rf)""",
            "example": {
                "question": "Suppose a stock has the following information. It is listed on the London stock exchange and operates throughout Europe. The yield on a UK 10 year treasury is 2.8%. The stock in question will earn 8.6% as per historical data. The Beta for the stock is 1.4, i.e., it is 140% volatile to the changes in the general stock market. What is the expected rate of return?",
                "solution": """def solution():
    # Given values. 
    rf = 0.028 # The yield on a UK 10 year treasury 
    beta = 1.4 # The stock is 140% volatile to the changes in the general stock market 
    rm = 0.086 # The stock in question will earn 8.6% as per historical data 
    # Calculate the expected return . 
    result = expected_return(rf, beta, rm) 
    # Return the result. 
    return result""",
                "answer": 0.109
            }
        }
    ]

    domain2book = {
        'causality': 'Introduction to Causal Inference',
        'physics': 'University Physics',
        'chemistry': "Atkins' Physical Chemistry",
    }
    book = domain2book[args.domain]
    structure = json.load(open(f'outputs/structure_{args.domain}.json'))
    
    data = []
    prompts = []
    for c in structure:
        for s in structure[c]:
            prompt = prompt_template.format(text=structure[c][s], book=book, chapter=c, example=json.dumps(example, indent=2))
            prompts.append(prompt)
            data.append({
                'chapter_name': c,
                'section_name': s,
                'prompt': prompt
            })
    
    all_responses = run_inference(prompts, output_folder, args)
            
    for idx, i in enumerate(data):
        if idx >= len(all_responses):
            break
        data[idx]['output'] = all_responses[idx]
            
    output_file = f'outputs/{args.model_name}_{args.domain}_tools_raw_output.json'
    json.dump(data, open(output_file, 'w'), indent = 4)
    
    # transform output to tools
    skills = []
    for idx, i in enumerate(data):
        output = i['output'].strip()
        cur = output[output.find('['):output.rfind(']')+1]
        try:
            cur = json.loads(cur, strict=False)
            for s in cur:
                skill = s.copy()
                skill['chapter_name'] = i['chapter_name']
                skill['section_name'] = i['section_name']
                skills.append(skill)
        except:
            continue
    print(len(skills))
    
    json.dump(skills, open(f'outputs/{args.model_name}_{args.domain}_tools_unfiltered.json', 'w'), indent=4)
    
    remove_tmp_files(output_folder)