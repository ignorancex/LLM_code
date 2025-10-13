"""This file is used to generate responses with single-topic or multiple-topic settings.
The generated responses are used for the facts exhaustion experiment."""

from openai import OpenAI

from datetime import datetime
from tqdm import tqdm
import argparse
import os
import logging
import time
from tools import *

# =============================================================================
#       Single-topic setting in facts exhaustion experiment                   #
# =============================================================================
_LENGTH_PLACEHOLDER = '[LENGTH]'
_TOPIC_1_PLACEHOLDER = '[TOPIC1]'
SINGLE_TOPIC_FACTUALITY_PROMPT = f"""
You are a helpful assistant. You will be given an entity name and one topic: `{_TOPIC_1_PLACEHOLDER}`. \
You need to generate a bio for the entity that relates to the topic. Here are the instructions: 
1. Generate a bio relates to "{_TOPIC_1_PLACEHOLDER}" with around {_LENGTH_PLACEHOLDER} words.
2. The response format should be like:
### {_TOPIC_1_PLACEHOLDER} ###
<Bio for {_TOPIC_1_PLACEHOLDER}>
3. Be sure to only include accurate, factual information in the response.
4. The bio should be comprehensive and detailed.
5. Do not include any controversial, disputable, or inaccurate factual claims in the response.
6. Return ONLY the bio, and nothing else.
""".strip()

# =============================================================================
#        Multiple-topic setting in facts exhaustion experiment                #
# =============================================================================
_TOPIC_1_PLACEHOLDER = '[TOPIC1]'
_TOPIC_2_PLACEHOLDER = '[TOPIC2]'
_TOPIC_1_LENGTH_PLACEHOLDER = '[TOPIC1_LENGTH]'
_TOPIC_2_LENGTH_PLACEHOLDER = '[TOPIC2_LENGTH]'

MULTIPLE_TOPIC_FACTUALITY_PROMPT = f"""
You are a helpful assistant. You will be given an entity name and two topics: `{_TOPIC_1_PLACEHOLDER}` \
and `{_TOPIC_2_PLACEHOLDER}`. 
You need to generate a bio for the entity that relates to the topics. Here are the instructions: 
1. Firstly generate a bio relates to "{_TOPIC_1_PLACEHOLDER}" with around {_TOPIC_1_LENGTH_PLACEHOLDER} words.
2. Then generate a bio relates to "{_TOPIC_2_PLACEHOLDER}" with around {_TOPIC_2_LENGTH_PLACEHOLDER} words.
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

def extract_evaluation_response(response, args):
    """Extract the evaluation response from the generated response."""
    if generic_abstain_detect(response):
        return None, None
    
    else:
        if args.setting == "single":
            topic1_response = extract_hash_block(response, 1)
            topic2_response = None
            eval_response = topic1_response
        
        elif args.setting == "multiple":
            topic1_response, topic2_response = extract_hash_block(response, 2)
            eval_response = topic1_response + "\n" + topic2_response
        
        return topic1_response, topic2_response, eval_response

def generate_bio(client, task, args, max_retries=5):
    
    if args.setting == "single":
        system_prompt = SINGLE_TOPIC_FACTUALITY_PROMPT.replace(_TOPIC_1_PLACEHOLDER, args.topic1)\
            .replace(_LENGTH_PLACEHOLDER, str(args.topic1_length))
    elif args.setting == "multiple":
        system_prompt = MULTIPLE_TOPIC_FACTUALITY_PROMPT.replace(_TOPIC_1_PLACEHOLDER, args.topic1)\
            .replace(_TOPIC_2_PLACEHOLDER, args.topic2)\
            .replace(_TOPIC_1_LENGTH_PLACEHOLDER, str(args.topic1_length))\
            .replace(_TOPIC_2_LENGTH_PLACEHOLDER, str(args.topic2_length))
    
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
            
            topic1_response, topic2_response, eval_response = \
                extract_evaluation_response(response, args)
            
            task['output'] = eval_response
            task['topic1_output'] = topic1_response
            task['topic2_output'] = topic2_response
            task['all_output'] = response
            return task
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Max retries exceeded for topic '{task['topic']}': {e}")
            time.sleep(1)

def main(args):
    
    if args.setting == "single":
        logging.info(f"Generating biographies with `{args.setting}-topic` setting. \
            with topic '{args.topic1}' around {args.topic1_length} words.")
    elif args.setting == "multiple":
        logging.info(f"Generating biographies with `{args.setting}-topic` setting. \
            with topic 1 '{args.topic1}' around {args.topic1_length} words \
            and topic 2 '{args.topic2}' around {args.topic2_length} words.")
    
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
    if args.setting == "single":
        output_path = f'{args.output_dir}/{args.model}_single_{map_to_name[args.topic1]}{args.topic1_length}_{dt_string}.jsonl'
    elif args.setting == "multiple":
        output_path = f'{args.output_dir}/{args.model}_multiple_{map_to_name[args.topic1]}{args.topic1_length}_{map_to_name[args.topic2]}{args.topic2_length}_{dt_string}.jsonl'
    
    for task in tqdm(tasks):
        result = generate_bio(client, task, args)
        jsonlines_dump(output_path, result)
    
    logging.info(f"All tasks completed. Results saved to {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate biographies for facts exhaustion experiment")
    parser.add_argument('--input_path', type=str, \
        default='../data/dataset/biography_generation.jsonl',\
        help="Path to the input JSONL file containing tasks")
    parser.add_argument('--output_dir', type=str, \
        default='output/facts_exhaustion',)
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help="OpenAI model to use for generation")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0)
    
    parser.add_argument('--setting', type=str, default="single",
                        choices=["single", "multiple"],
                        help="Setting for facts exhaustion: 'single' for single topic, 'multiple' for two topics")

    parser.add_argument('--topic1', type=str, default='personal life',
                        help="Topic for context section of biography")
    parser.add_argument('--topic2', type=str, default='career',
                        help="Topic for evaluation section of biography")
    
    parser.add_argument('-t1l', '--topic1_length', type=int, default=200, 
                        help="Length for topic 1 section")
    parser.add_argument('-t2l', '--topic2_length', type=int, default=200, 
                        help="Length for topic 2 section")
    parser.add_argument('--api_key', type=str, required=True,
                        help="OpenAI API key")
    
    args = parser.parse_args()
    main(args)
