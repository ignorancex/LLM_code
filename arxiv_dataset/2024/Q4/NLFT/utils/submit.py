import argparse
import json
import os
import re
import warnings

import torch
import transformers
import pandas as pd

from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer

from helper import clean_answer, extract_answer_from_output, sys, clear_gpu_memory

clear_gpu_memory()

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
    conversation = chatbot(messages, temperature=0.6, max_new_tokens=512) # temperature=0.6，其他是 0.3
    output = conversation[0]['generated_text'][-1]["content"]
    return output


def main():
    parser = argparse.ArgumentParser(description='evaluate finetuned LLM on gsm8k testset for accuracy (sft, nlft)')
    parser.add_argument('--input_json', type=str, required=True, help='path to gsm8k testset json')
    parser.add_argument('--llama3_path', type=str, required=True, help='path to finetuned model (merged)')
    parser.add_argument('--prev_count', type=int, default=0, help='previous correct count for evaluation') # 默认不填
    parser.add_argument('--prev_sum', type=int, default=0, help='previous sum count for evaluation') # 默认不填
    parser.add_argument('--max_sum', type=int, default=-1, help='max sum count for evaluation') # 默认不填

    args = parser.parse_args()

    file_path = args.input_json
    load_llama3_model(args.llama3_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查是否有'problem'和'answer'这两列
    count = args.prev_count
    sum = args.prev_sum
    max_sum = args.max_sum if args.max_sum != -1 else len(data)

    for idx, entry in tqdm(enumerate(data), total=len(data), desc="Evaluating", unit="item"):
        if idx < sum:
            continue
        if idx > max_sum:
            break
        query = entry['question']
        response = entry['answer']
        ans = query_ans(query)

        llm_answer = clean_answer(ans)
        std_answer = extract_answer_from_output(response)
        print(f"LLM Answer: {llm_answer}, Standard Answer: {std_answer}")
        if llm_answer == std_answer:
            count += 1
        else:
            print("LLM gives wrong answer:")
            print(ans)
        sum += 1
        p = count/sum
        print(f"正确率：{count}/{sum}",p)
    print("CORRECT SUM:",count)


if __name__ == '__main__':
    main()
