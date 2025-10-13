"""This file is used to generate responses with model's dafault output length."""
from openai import OpenAI

from datetime import datetime
from tqdm import tqdm
import argparse
import os
import logging
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import *

NAIVE_FACTUALITY_PROMPT = f"""
You are a helpful assistant. You will be given an entity name. You need to generate a bio for it. Here are the instructions:
1. Be sure to only include accurate, factual information in the response.
2. The bio should be comprehensive and detailed.
3. Do not include any controversial, disputable, or inaccurate factual claims in the response.
4. Return ONLY the bio, and nothing else.
""".strip()
    
def generate_bio(client, task, args, max_retries=5):
    """Generate a biography for a given topic."""
    
    system_prompt = NAIVE_FACTUALITY_PROMPT
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
    
    # check if output directory exists, if not, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = f'{args.output_dir}/{args.model}_default_{dt_string}.jsonl'
    
    for task in tqdm(tasks):
        result = generate_bio(client, task, args)
        jsonlines_dump(output_path, result)
    
    logging.info(f"All tasks completed. Results saved to {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate biographies for autocorrelation analysis")
    parser.add_argument('--input_path', type=str, \
        default='../../data/dataset/biography_generation.jsonl',\
        help="Path to the input JSONL file containing tasks")
    parser.add_argument('--output_dir', type=str, \
        default='../output/error_propagation')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help="OpenAI model to use for generation")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--api_key', type=str, required=True,
                        help="OpenAI API key")
    
    args = parser.parse_args()
    main(args)