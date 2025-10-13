import json
import argparse
import os
import sys
import openai
from tqdm import tqdm
from io import StringIO

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    args = parser.parse_args()

    output_folder = f'outputs/tmp_response/'
    tools = json.load(open(f'outputs/{args.model_name}_{args.domain}_tools_unfiltered_feedback.json', 'r'))
    
    revision_template = """Please revise the skill according to the feedback. 
The skill is a python function with comments of parameters and returns, accompanied by a description and a demonstration example of using the skill. Please try to keep the original intent of the skill, and modify the description/function/example to address the feedback.
Note the indent in code is 4 spaces. All packages used should be imported inside the function. The function should be self-contained. The answer to the example question is encouraged to be numerical.
NOTE THAT THE SKILL PYTHON CODE SHOULD NOT BE SPECIFIC TO/ONLY APPLIED TO THE CHOSEN EXAMPLE! PLEASE GENERATE GENERAL SKILL CODE.
The output should be in *complete* json structure as the original skill, starting with '{' and ending with '}'.

Original Skill:
{tool}

Feedback:
{feedback}
"""
    
    prompts = []
    for i in tools:
        if i['feedback'] != 'Succeed!':
            tool = {}
            tool['description'] = i['description']
            tool['function'] = i['function']
            tool['example'] = i['example']
            prompt = revision_template.replace('{tool}', json.dumps(tool, indent=4)).replace('{feedback}', i['feedback'])
            prompts.append(prompt)

    all_responses = run_inference(prompts, output_folder, args)
        
    response_idx = 0
    for idx, i in enumerate(tools):
        if i['feedback'] == 'Succeed!':
            continue
        output = all_responses[response_idx].strip()
        response_idx += 1
        cur = output[output.find('{'):output.rfind('}')+1]
        try:
            tool = json.loads(cur, strict=False)
            tool['chapter_name'] = i['chapter_name']
            tool['section_name'] = i['section_name']
            tool['feedback'] = 'Refined.'
            tools[idx] = tool
        except Exception as e:
            print(e)
    assert response_idx == len(all_responses)

    json.dump(tools, open(f'outputs/{args.model_name}_{args.domain}_tools_refined.json', 'w'), indent=4)
    
    remove_tmp_files(output_folder)