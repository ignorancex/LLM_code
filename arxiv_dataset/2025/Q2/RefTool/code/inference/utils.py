import os
import json
import re
import math
import multiprocessing
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import openai
import google.generativeai as genai
import anthropic
from typing import Union
import torch

seed = 42

def model_name_to_path(model_name):
    if model_name == 'gpt4':
        model = 'gpt-4-1106-preview' 
    elif model_name == 'gpt4o':
        model = 'gpt-4o-2024-11-20' 
    elif model_name == 'gemini-pro':
        model = 'gemini-1.5-pro-002'
    elif model_name == 'llama3.1-70b':
        model = ""  # the model path
    elif model_name == 'gpt3.5':
        model = 'gpt-3.5-turbo-0125'
    return model


def get_results_api(data_slice, cpu_id, model_name, output_folder, max_tokens = 512):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_file = os.path.join(output_folder, f'response_{cpu_id}.json')
    
    model = model_name_to_path(model_name)
        
    if 'gemini' in model:
        config = genai.GenerationConfig(candidate_count=1,
                      max_output_tokens = max_tokens,
                      temperature = 0.0)
        gemini_model = genai.GenerativeModel(model, generation_config = config)
    elif 'llama' in model.lower():
        client = openai.OpenAI(api_key="EMPTY", base_url="")  # llama api url

    ret = []
    for i in tqdm(range(len(data_slice)) , desc=f'{cpu_id}'):
        input_text = data_slice[i]
        temp = 3
        while temp > 0:
            try:
                if 'gpt' in model:
                    completion = openai.chat.completions.create(
                        model = model,
                        messages = [{'role' : 'user', 'content' : input_text}],
                        temperature = 0.0,
                        max_tokens = max_tokens,
                        seed=seed
                    )
                    answer = completion.choices[0].message.content
                
                elif 'gemini' in model:
                    completion = gemini_model.generate_content(input_text)
                    answer = completion.text.strip()
                
                elif 'llama' in model.lower():
                    completion = client.chat.completions.create(
                        model = model,
                        max_tokens = max_tokens,
                        temperature = 0.0,
                        messages = [{'role' : 'user', 'content' : input_text}],
                        stream=False,
                        seed=seed
                    )
                    answer = completion.choices[0].message.content
                    
                ret.append(answer)
                break
            except Exception as e:
                temp -= 1
                if temp == 0:
                    print(e)
                
        if temp > 0:
            continue
        else:
            ret.append('None')

    json.dump(ret, open(output_file, 'w'), indent=4)
    return None


def run_inference(prompts, output_folder, args):
    import multiprocessing
    
    pool_num = 16

    pool = multiprocessing.Pool(pool_num)
    results_unmerged = []
    itv = 10

    for sid in list(range(0, len(prompts), itv)):
        results_unmerged.append(
            pool.apply_async(get_results_api, (prompts[sid:sid+itv], sid, args.model_name, output_folder, args.max_tokens))
        )

    pool.close()
    pool.join()

    ### merge all the response in output_folder
    all_files = os.listdir(output_folder)
    all_files = [os.path.join(output_folder, file) for file in all_files if '.json' in file and 'response_' in file]
    all_files.sort(key=lambda i: int(i.split('response_')[1].split('.json')[0]))

    all_responses = []
    for file in all_files:
        all_responses.extend(json.load(open(file)))

    return all_responses


def remove_tmp_files(output_folder):
    all_files = os.listdir(output_folder)
    all_files = [os.path.join(output_folder, file) for file in all_files if '.json' in file and 'response_' in file]
    for file in all_files:
        os.remove(file)
        
        
def remove_not(x):
    match_number = re.compile('[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?')
    result=re.findall(match_number, x)
    if len(result) !=0:
        return re.split(match_number, x)[-1]
    return None


def get_table_input(data_files, num_of_rows, data_dir='../../evaluation_data/data/'):
    output = []
    for file in data_files:
        df = pd.read_csv(os.path.join(data_dir, file))
        ## shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        output.append(f'{file}:\n{df.head(num_of_rows).to_markdown()}')
    return ('\n\n'.join(output))


def get_pot_input_qrdata(entry, num_of_rows = 10):
    instruction = '''You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the provided data. The data description, first 10 rows of the data, and the question can be found below. Please write python code to analyze the whole data and answer the question. Please encase the Python code within triple backticks. You can use any python library you imported. The returned value of the code is supposed to be the answer. The format of the code should be
```python
def solution():
    # import libraries if needed

    # load data
    
    # write code to get the answer
    
    # return answer
```
'''
    question = entry['question']
    data_description = entry['data_description']

    while num_of_rows > 0:
        prompt = f'''
Data Description:
{data_description}

First {num_of_rows} rows of the data:
{get_table_input(entry['data_files'], num_of_rows)}

Question:
{question}

Response:
```python
'''
        if len(prompt.split()) < 3000:
            break
        num_of_rows -= 1

    return instruction + prompt


def get_pot_input_theoremqa(entry):
    instruction = '''You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question below. Please write python code to answer the question. Please encase the Python code within triple backticks. You can use any python library you imported. The returned value of the code is supposed to be the answer. The format of the code should be
```python
def solution():
    # import libraries if needed

    # write code to get the answer
    
    # return answer
```
'''
    problem_text = entry["Question"]

    prompt = f'''Question:
{problem_text}

Response:
'''
    return instruction + prompt


def get_pot_input_scibench(entry):
    instruction = '''You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question below. Please write python code to answer the question. Please encase the Python code within triple backticks. You can use any python library you imported. The returned value of the code is supposed to be the answer. The format of the code should be
```python
def solution():
    # import libraries if needed

    # write code to get the answer
    
    # return answer
```
'''
    unit_prob = entry["unit"]
    if remove_not(entry["unit"]):
        unit_prob = remove_not(entry["unit"])
    problem_text = entry["problem_text"] + " The unit of the answer is " + unit_prob + "."

    prompt = f'''Question:
{problem_text}

Response:
'''
    return instruction + prompt