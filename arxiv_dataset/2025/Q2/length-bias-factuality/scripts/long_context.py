"""This file is used to generate responses with both context section and evaluation section lengths."""
from openai import OpenAI

from datetime import datetime
from tqdm import tqdm
import argparse
import os
import logging
import time
from tools import *


_TOPIC_1_PLACEHOLDER = '[TOPIC1]'
_TOPIC_2_PLACEHOLDER = '[TOPIC2]'
_CONTEXT_LENGTH_PLACEHOLDER = '[CONTEXT_LENGTH]'
_EVALUATION_LENGTH_PLACEHOLDER = '[EVALUATION_LENGTH]'

LONG_CONTEXT_PROMPT = f"""
You are a helpful assistant. You will be given an entity name and two topics: `{_TOPIC_1_PLACEHOLDER}` \
and `{_TOPIC_2_PLACEHOLDER}`. 
You need to generate a bio for the entity that relates to the topics. Here are the instructions: 
1. Firstly generate a bio relates to "{_TOPIC_1_PLACEHOLDER}" with around {_CONTEXT_LENGTH_PLACEHOLDER} words.
2. Then generate a bio relates to "{_TOPIC_2_PLACEHOLDER}" with around {_EVALUATION_LENGTH_PLACEHOLDER} words.
3. The response format should be like:
### {_TOPIC_1_PLACEHOLDER} ###
<Bio for {_TOPIC_1_PLACEHOLDER}>

### {_TOPIC_2_PLACEHOLDER} ###
<Bio for {_TOPIC_2_PLACEHOLDER}>
4. Be sure to only include accurate, factual information in the response.
5. The bio should be comprehensive and detailed.
6. Do not include any controversial, disputable, or inaccurate factual claims in the response.
7. Return ONLY the bio, and nothing else.
""".strip()

def split_evaluation_section(response):
    if generic_abstain_detect(response):
        return None, None
    else:
        context_response, evaluation_response = extract_hash_block(response, 2)
        return context_response, evaluation_response

def generate_bio(client, task, args, max_retries=5):
    
    system_prompt = LONG_CONTEXT_PROMPT.replace(_TOPIC_1_PLACEHOLDER, args.topic1)\
        .replace(_TOPIC_2_PLACEHOLDER, args.topic2)\
        .replace(_CONTEXT_LENGTH_PLACEHOLDER, str(args.context_length))\
        .replace(_EVALUATION_LENGTH_PLACEHOLDER, str(args.evaluation_length))
    
    question = f"Tell me a bio of {task['topic']}."

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
            response = completion.choices[0].message.content
            context_response, evaluation_response = split_evaluation_section(response)
            task['output'] = evaluation_response
            task['topic1_output'] = context_response
            task['all_output'] = response
            return task
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Max retries exceeded for topic '{task['topic']}': {e}")
            time.sleep(1)

def main(args):
    
    assert args.context_length > 0, "Context length must be greater than 0"
    logging.info(f"Generating biographies with context topics '{args.topic1}' with length {args.context_length} \
        and evaluation topic '{args.topic2}' with length {args.evaluation_length} using model {args.model}.")
    
    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    client = OpenAI(api_key=args.api_key)
    
    all_data = jsonlines_load(args.input_path)
    tasks = all_data[args.start:args.end]
    
    # check if output directory exists, if not, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    map_to_name={
        "early life": "er",
        "career": "cr",
        "personal life": "pr",
    }
    output_path = f'{args.output_dir}/{args.model}_{map_to_name[args.topic1]}{args.context_length}_{map_to_name[args.topic2]}{args.evaluation_length}_{dt_string}.jsonl'
    
    for task in tqdm(tasks):
        result = generate_bio(client, task, args)
        jsonlines_dump(output_path, result)
    
    logging.info(f"All tasks completed. Results saved to {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate biographies for long context experiments")
    parser.add_argument('--input_path', type=str, \
        default='../data/dataset/biography_generation.jsonl',\
        help="Path to the input JSONL file containing tasks")
    parser.add_argument('--output_dir', type=str, \
        default='output/long_context')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help="OpenAI model to use for generation")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0)

    parser.add_argument('--topic1', type=str, default='personal life',
                        help="Topic for context section of biography")
    parser.add_argument('--topic2', type=str, default='career',
                        help="Topic for evaluation section of biography")
    
    parser.add_argument('-cl', '--context_length', type=int, default=200, 
                        help="Length of the context section")
    parser.add_argument('-el', '--evaluation_length', type=int, default=200, 
                        help="Length of the evaluation section")
    parser.add_argument('--api_key', type=str, required=True,
                        help="OpenAI API key")
    
    args = parser.parse_args()
    main(args)
