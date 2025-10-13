"""This file is used to generate responses with varying lengths."""
from openai import OpenAI

from datetime import datetime
from tqdm import tqdm
import argparse
import os
import logging
import time
from tools import *

_LENGTH_PLACEHOLDER = '[LENGTH]'
_CAT_PLACEHOLDER = '[CAT]'

LENGTH_BIOGRPHY_PROMPT = f"""
You are a helpful assistant. You will be given an entity name. You need to generate a bio for it. \
Here are the instructions:
1. The bio should be around {_LENGTH_PLACEHOLDER} words.
2. Be sure to only include accurate, factual information in the response.
3. The bio should be comprehensive and detailed.
4. Do not include any controversial, disputable, or inaccurate factual claims in the response.
5. Return ONLY the bio, and nothing else.
""".strip()

LENGTH_LONG_FACT_PROMPT = f"""
You are a helpful assistant. You will be given an entity related to `{_CAT_PLACEHOLDER}`. You need to provide a description of it. \
Here are the instructions:
1. The response should be around {_LENGTH_PLACEHOLDER} words.
2. Be sure to only include accurate, factual information in the response.
3. The response should be comprehensive and detailed.
4. Do not include any controversial, disputable, or inaccurate factual claims in the response.
5. Return ONLY the information about the entity, and nothing else.
6. Return the information in paragraph form using plain text, not in markdown or any other format.
""".strip()
    
def generate_bio(client, task, task_type, args, max_retries=5):
    """Generate a response for a given topic with a requested output length."""
    
    if task_type == "biography":
        system_prompt = LENGTH_BIOGRPHY_PROMPT.replace(_LENGTH_PLACEHOLDER, str(args.length))
        question = f"Tell me a bio of {task['topic']}."
        
    elif task_type == "long_fact":
        system_prompt = LENGTH_LONG_FACT_PROMPT.replace(_LENGTH_PLACEHOLDER, str(args.length))\
            .replace(_CAT_PLACEHOLDER, task['cat'])
        
        question = f"Tell me about {task['topic']}."    
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
            )
            task['input'] = question
            task['output'] = completion.choices[0].message.content
            return task
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Max retries exceeded for topic '{task['topic']}': {e}")
            time.sleep(1)

def main(args):
    
    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    client = OpenAI(api_key=args.api_key)
    
    all_data = jsonlines_load(args.input_path)
    tasks = all_data[args.start:args.end]
    
    task_type = ""
    
    if "biography" in args.input_path:
        task_type = "biography"
        logging.info(f"Processing biography generation tasks from {args.input_path}")
    elif "long_fact" in args.input_path:
        task_type = "long_fact"
        logging.info(f"Processing long fact generation tasks from {args.input_path}")
    else:
        logging.error(f"Unknown task type in {args.input_path}. Please check the input file.")
        return
    
    # check if output directory exists, if not, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = f'{args.output_dir}/{args.model}_{task_type}_len{args.length}_{dt_string}.jsonl'
    
    for task in tqdm(tasks):
        result = generate_bio(client, task, task_type, args)
        jsonlines_dump(output_path, result)
    
    logging.info(f"All tasks completed. Results saved to {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate biographies with varying lengths")
    parser.add_argument('--input_path', type=str, \
        default='../data/dataset/biography_generation.jsonl',\
        help="Path to the input JSONL file containing tasks")
    parser.add_argument('--output_dir', type=str, \
        default='output/length_bias')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help="OpenAI model to use for generation")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--length', type=int, default=100, 
                        help="Length of the biography to generate")
    parser.add_argument('--api_key', type=str, required=True,
                        help="OpenAI API key")
    
    args = parser.parse_args()
    main(args)
