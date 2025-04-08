import argparse
import json
import os
import re
import warnings

import torch
import transformers
import pandas as pd
from openai import OpenAI

from helper import clean_answer, extract_answer_from_output, sys, sys2

api_key = os.environ['OPENAI_API_KEY']
api_base = os.environ['OPENAI_API_BASE']
client = OpenAI(api_key=api_key, base_url=api_base)


def gpt_api(sys_prompt, usr_prompt):
    global client
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": sys_prompt},
                {"role": "user",
                 "content": usr_prompt}
            ],
            max_tokens=512,
        )
        response = f"{completion.choices[0].message.content}"
    except Exception as e:
        # Returning a string indicating an error occurred
        print(f"Error: {str(e)}")
        raise
        # return f"Error: {str(e)}"
    return response

def query_ans(inp):
    usr = f"### Question: {inp}"
    response = gpt_api(sys, usr)
    return response


def query_why(query, ans, resp):
    usr = f"### Question: {query}\n\n### Wrong Answer: {ans}\n\n### Correct Answer: {resp}\n"
    response = gpt_api(sys2, usr)
    return response


def main():
    parser = argparse.ArgumentParser(description='give judgment from GPT-4')
    parser.add_argument('--input_json', type=str, required=True, help='path to input json file') # default: gsm8k_train_cot_llama3_8b.json
    parser.add_argument('--output_json', type=str, default=None, help='path to output json file') # default: gsm8k_train_cot_gpt4_judgment_llama3_8b.json
    parser.add_argument('--start_idx', type=int, default=0, help='start index in input json')
    parser.add_argument('--end_idx', type=int, default=-1, help='end index in input json')

    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    assert start_idx < end_idx
    chunk_size = 50

    output_json = './gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_{}_{}.json'.format(start_idx, end_idx) if args.output_json is None else args.output_json
    with open(output_json, 'a', encoding='utf-8') as outfile:
        outfile.write('[\n')
        # fixed question, answer, ans -> generate judgment

        for idx, item in enumerate(data):
            if idx >= end_idx:
                continue
            if idx < start_idx:
                continue
            query = item['question']
            response = item['answer']

            ans = item['ans']

            ans_number = clean_answer(ans)
            response_number = extract_answer_from_output(response)

            if ans_number != response_number:
                judgment = query_why(query, ans, response)
                if "response is correct" not in judgment:
                    item['judgment'] = judgment
                    print(ans)
                    print(response)
                    print(judgment)
                else:
                    item['judgment'] = None
                    print(f"answer:{ans_number},correct:{response_number}")
            else:
                item['judgment'] = None
                print(f"answer:{ans_number},correct:{response_number}")

            json.dump(item, outfile, ensure_ascii=False, indent=4)
            if idx < len(data) - 1:
                outfile.write(',\n')
            if (idx + 1) % chunk_size == 0:
                outfile.flush()
                print(f"[Checkpoint] Proceesed {idx + 1} items")
            print(idx)

        outfile.write('\n]')

if __name__ == '__main__':
    main()
