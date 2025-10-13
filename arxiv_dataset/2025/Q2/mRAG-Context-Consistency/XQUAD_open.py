import os
#cache_dir = os.getenv("TMPDIR")
cache_dir = "/projects/prjs1335/cache/"

from datasets import load_dataset
import torch
import jsonlines
import argparse
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from random import sample
import re

from vllm import LLM as VLLM
from vllm import SamplingParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

langs = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh']

def make_prompt(tokenizer, mname, instruction, content):
    if mname in ["CohereForAI/aya-expanse-8b", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]:
        messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": content}
                ]
    elif mname == "google/gemma-2-9b-it":
        messages = [{"role": "user", "content": instruction + " " + content}]
    else:
        raise ValueError("Not supported model.")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def contains_option(option_letter, response):
    #pattern = r'\b[A-D]\b(?!\s)'
    pattern = r'\b[A-D]\b'
    match = re.search(pattern, response)
    if not match:
        return False
    else:
        if option_letter == match.group(0): return True
        else: return False

def check_substring(gold_ans_list, response):
    res = False
    for ans in set(gold_ans_list):
        if ans.lower() in response.lower():
            res = True
            break
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mname", type=str, default="CohereForAI/aya-expanse-8b", help="LLM name on HF")
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size")

    args = parser.parse_args()

    task = "XQUAD"
    num_retrieval = 4
    save_dir = f"results/{task}_open/"

    # Load data
    all_data = dict()
    for lang in langs:
        all_data[lang] = load_dataset("google/xquad", f"xquad.{lang}", split="validation", cache_dir=cache_dir)
    
    data = all_data[args.lang]

    with open("instruction_open.json", encoding='utf-8') as f:
        instructions = json.load(f)
    f.close()

    tokenizer = AutoTokenizer.from_pretrained(args.mname, cache_dir=cache_dir)
    extra_kw = {"download_dir": cache_dir}
    vmodel = VLLM(
            model=args.mname,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.65,
            dtype=torch.bfloat16,
            distributed_executor_backend="mp",
            trust_remote_code=True,
            max_num_seqs=100,
            seed=2024,
            disable_custom_all_reduce=True,
            **extra_kw
            )
    
    num_ins = len(data)
    batch_prompts = []
    save_data = []
    start_id = 0

    for ins_id, ins in enumerate(tqdm(data)):
        meta_info = {"index": ins_id}
        prompts = []

        #doc_template = "Document [{ID}](Title: {T}): {P}\n"
        doc_template = "[{ID}]({T}): {P}\n"
        
        # (1) non context
        #content_non_ctx = "Question: " + ins['question'] + " Answer:"
        content_non_ctx = f"{ins['question']}\n"
        prompt_non_ctx = make_prompt(tokenizer, args.mname, instructions[args.lang]['non_ctx'], content_non_ctx)
        prompts.append(prompt_non_ctx)

        # (2) Different combinations
        for lang in langs:
            content_ctx = f"{all_data[lang][ins_id]['context']}\n{ins['question']}\n"
            prompt_ctx = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_ctx)
            prompts.append(prompt_ctx)
            meta_info[f'answer_text_{lang}'] = all_data[lang][ins_id]['answers']['text']

        save_data.append(meta_info)

        # Gather all prompts into a list
        num_prompts = len(prompts)
        batch_prompts += prompts
        
        if ((ins_id+1) % args.batch_size == 0) or ((ins_id+1) == num_ins):
            # Generate responses
            sampling_params = SamplingParams(
                    temperature=1.0,
                    top_k=1,
                    max_tokens=100,
                )
            #prompt_logprobs=0,
        
            seg_raw_responses_list = vmodel.generate(batch_prompts, sampling_params, use_tqdm=True)
            #print(seg_raw_responses_list)
            
            # Obtain responses for each prompt
            for raw_response_id, raw_response in enumerate(seg_raw_responses_list):
                current_id = start_id+int(raw_response_id/num_prompts)
                if raw_response_id % len(prompts) == 0:
                    field_prompt = 'prompt_non_ctx'
                    field_response = 'response_non_ctx'
                    field_correct = 'correct_non_ctx'
                    field_correct_out = 'correct_non_ctx_out' # useless term, only for placeholder
                    outlang = args.lang # useless term, only for placeholder
                else:
                    outlang = langs[(raw_response_id % len(prompts))-1]
                    field_prompt = f"prompt_ctx_{outlang}"
                    field_response = f"response_ctx_{outlang}"
                    field_correct = f"correct_ctx_{outlang}"
                    field_correct_out = f"correct_ctx_{outlang}_out"

                save_data[current_id][field_prompt] = raw_response.prompt
                save_data[current_id][field_response] = raw_response.outputs[0].text
                save_data[current_id][field_correct] = check_substring(save_data[current_id][f'answer_text_{args.lang}'], raw_response.outputs[0].text)
                save_data[current_id][field_correct_out] = check_substring(save_data[current_id][f'answer_text_{outlang}'], raw_response.outputs[0].text)

            batch_prompts = []
            start_id = start_id + int(len(seg_raw_responses_list)/num_prompts)
            #start_id = ins_id+1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}{args.mname.split('/')[-1]}_{args.lang}.json", 'w', encoding='utf-8') as f:
        #json.dump(save_data, f, indent=4, sort_keys=False)
        json.dump(save_data, f, sort_keys=False)
    print(len(save_data))

if __name__ == "__main__":
    main()
