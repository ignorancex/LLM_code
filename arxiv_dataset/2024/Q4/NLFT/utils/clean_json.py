import argparse
import json
import os
import re
import warnings

import torch
import transformers
import pandas as pd

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer

from helper import clean_answer, extract_answer_from_output, sys, sys2

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_DATASETS_CACHE'] = r"/root/tmp/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

model_llama3 = None
tokenizer = None

def load_llama3_model(llama3_path):
    global model_llama3, tokenizer
    model_llama3 = AutoModelForCausalLM.from_pretrained(
        llama3_path,
        low_cpu_mem_usage=True,
        # load_in_4bit=True,
        device_map={"": Accelerator().local_process_index},
    )
    tokenizer = AutoTokenizer.from_pretrained(llama3_path)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    model_llama3.resize_token_embeddings(model_llama3.config.vocab_size + 1)

def query_ans(inp):
    usr = inp
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    chatbot = transformers.pipeline(
        "text-generation",
        model=model_llama3,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    conversation = chatbot(messages, temperature=0.3, max_new_tokens=512)
    output = conversation[0]['generated_text'][-1]["content"]
    return output


def query_why(query, ans, resp):
    usr = f"### Question: {query}\n### Wrong Answer: {ans}\n### Correct Answer: {resp}"
    messages = [{"role": "system", "content": sys2}, {"role": "user", "content": usr}]
    chatbot = transformers.pipeline(
        "text-generation",
        model=model_llama3,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    conversation = chatbot(messages, temperature=0.3, max_new_tokens=512)
    output = conversation[0]['generated_text'][-1]["content"]
    return output

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='generate output using llama3-8b or llama3-8b-instruct')
    parser.add_argument('--llama3_path', type=str, required=True, help='path to llama3-8b OR llama3-8b-instruct')
    parser.add_argument('--input_json', type=str, default='./gsm8k_train.json', help='path to input json')
    parser.add_argument('--output_json', type=str, default=None, help='path to output json') # default: ./gsm8k_train_cot_llama3_8b.json
    parser.add_argument('--start_idx', type=int, required=True, help='start index in input json')
    parser.add_argument('--end_idx', type=int, required=True, help='end index in input json')
    parser.add_argument('--judgment_enabled', type=str2bool, nargs='?', const=True, default=True, help='self generating judgment')

    args = parser.parse_args()

    load_llama3_model(args.llama3_path)

    start_idx = args.start_idx
    end_idx = args.end_idx
    chunk_size = 50

    with open(args.input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    assert len(data) == 7473 # gsm8k training set
    output_json = args.output_json if args.output_json else 'gsm8k_train_cot_llama3_8b_instruct_{}_{}.json'.format(start_idx, end_idx)
    judgment_enabled = args.judgment_enabled
    with open(output_json, 'a', encoding='utf-8') as outfile:
        outfile.write('[\n')

        for idx, item in enumerate(data):
            if idx >= end_idx:
                continue
            if idx < start_idx:
                continue
            query = item['question'] # gsm8k column name: question
            response = item['answer'] # gsm8k column name: answer

            ans = query_ans(query)

            item['ans'] = ans

            ans_number = clean_answer(ans)
            response_number = extract_answer_from_output(response)

            if ans_number != response_number:
                if judgment_enabled:
                    judgment = query_why(query, ans, response)
                    if "The response is correct" not in judgment: # 与 sys2 prompt 对应
                        item['judgment'] = judgment
                        print(judgment)
                    else:
                        item['judgment'] = None
                else:
                    item['judgment'] = 'TODO: waiting for GPT4 to generate judgment'
            else:
                item['judgment'] = None

            # 将结果写入文件
            json.dump(item, outfile, ensure_ascii=False, indent=4)
            # 如果不是最后一个元素，添加逗号和换行符
            if idx < len(data) - 1:
                outfile.write(',\n')
            if (idx + 1) % chunk_size == 0:
                outfile.flush()
                print(f"[Checkpoint] Proceesed {idx + 1} items")
            print(idx)

        outfile.write('\n]')

if __name__ == '__main__':
    main()
