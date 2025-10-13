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

#langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
#langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
#langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']

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

    task = 'GMMLU'
    num_retrieval = 4
    # Load data
    load_path = f"data_inlang_{task}_filter/{args.lang}_gtr_top30.json"
    with open(load_path, encoding='utf-8') as f:
        data = json.load(f)
    f.close()

    save_dir = f"results/{task}_open/"

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
        # Prepare inlang gold docs and outlang gold docs
        with open(f"data_inlang_{task}_filter/golddoc/{ins['index']}.jsonl") as f:
            gold_docs = f.read()
            gold_docs = gold_docs.split('\n')[:-1]
            gold_docs = [json.loads(i) for i in gold_docs]
        f.close()

        # filter inlang and outlang gold docs
        gold_docs_inlang = []
        gold_docs_outlang = []
        for gd in gold_docs:
            if gd['lang'] == args.lang:
                gold_docs_inlang.append(gd)
            else:
                gold_docs_outlang.append(gd)
        
        if gold_docs_inlang and gold_docs_outlang:
            #doc_template = "Document [{ID}](Title: {T}): {P}\n"
            doc_template = "[{ID}]({T}): {P}\n"
            
            meta_info = {"index": ins['index'], "answer_text": [ins['answer_text']]}
            prompts = []

            # (1) non context
            #content_non_ctx = "Question: " + ins['question'] + " Answer:"
            content_non_ctx = f"{ins['question']}\n"
            prompt_non_ctx = make_prompt(tokenizer, args.mname, instructions[args.lang]['non_ctx'], content_non_ctx)
            prompts.append(prompt_non_ctx)

            # (2) inlang gold passage
            if len(gold_docs_inlang) >= 3:
                inlang_gold_docs = sample(gold_docs_inlang, 3)
            elif len(gold_docs_inlang) == 2:
                inlang_gold_docs = sample(gold_docs_inlang, 2) + sample(gold_docs_inlang, 1)
            elif len(gold_docs_inlang) == 1:
                inlang_gold_docs = sample(gold_docs_inlang, 1) + sample(gold_docs_inlang, 1) + sample(gold_docs_inlang, 1)

            for _inlang_gold_docs in inlang_gold_docs:
                context_ctx_inlang = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate([_inlang_gold_docs])]
                #content_ctx_inlang = "".join(doc_top_inlang) + "Question: " + ins['question'] + " Answer:"
                content_ctx_inlang = f"{''.join(context_ctx_inlang)}{ins['question']}\n"
                prompt_ctx_inlang = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_ctx_inlang)
                prompts.append(prompt_ctx_inlang)

            # (3) outlang gold passage
            if len(gold_docs_outlang) >= 3:
                outlang_gold_docs = sample(gold_docs_outlang, 3)
            elif len(gold_docs_outlang) == 2:
                outlang_gold_docs = sample(gold_docs_outlang, 2) + sample(gold_docs_outlang, 1)
            elif len(gold_docs_outlang) == 1:
                outlang_gold_docs = sample(gold_docs_outlang, 1) + sample(gold_docs_outlang, 1) + sample(gold_docs_outlang, 1)

            for id_outlang_gold_docs, _outlang_gold_docs in enumerate(outlang_gold_docs):
                context_ctx_outlang = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate([_outlang_gold_docs])]
                #content_ctx_inlang = "".join(doc_top_inlang) + "Question: " + ins['question'] + " Answer:"
                content_ctx_outlang = f"{''.join(context_ctx_outlang)}{ins['question']}\n"
                prompt_ctx_outlang = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_ctx_outlang)
                prompts.append(prompt_ctx_outlang)
                
                meta_info[f'answer_text_out_{id_outlang_gold_docs}'] = [_outlang_gold_docs['answer']]
                meta_info[f'lang_text_out_{id_outlang_gold_docs}'] = _outlang_gold_docs['lang']

            save_data.append(meta_info)

            # Gather all prompts into a list
            num_prompts = len(prompts)
            batch_prompts += prompts
        
        if (((ins_id+1) % args.batch_size == 0) or ((ins_id+1) == num_ins)) and batch_prompts:
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
                remainder = (raw_response_id % len(prompts))
                if remainder == 0:
                    field_prompt = 'prompt_non_ctx'
                    field_response = 'response_non_ctx'
                    field_correct = 'correct_non_ctx'
                elif remainder in [1,2,3]:
                    field_prompt = f"prompt_ctx_inlang_{remainder-1}"
                    field_response = f"response_ctx_inlang_{remainder-1}"
                    field_correct = f"correct_ctx_inlang_{remainder-1}"
                elif remainder in [4,5,6]:
                    field_prompt = f"prompt_ctx_outlang_{remainder-4}"
                    field_response = f"response_ctx_outlang_{remainder-4}"
                    field_correct = f"correct_ctx_outlang_{remainder-4}"
                    field_correct_out = f"correct_ctx_outlang_out_{remainder-4}"
                else:
                    raise ValueError('Number of prompts does not match. Please double check.')

                save_data[current_id][field_prompt] = raw_response.prompt
                save_data[current_id][field_response] = raw_response.outputs[0].text
                save_data[current_id][field_correct] = check_substring(save_data[current_id]['answer_text'], raw_response.outputs[0].text)
                if remainder in [4,5,6]:
                    save_data[current_id][field_correct_out] = check_substring(save_data[current_id][f'answer_text_out_{remainder-4}'], raw_response.outputs[0].text)

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
