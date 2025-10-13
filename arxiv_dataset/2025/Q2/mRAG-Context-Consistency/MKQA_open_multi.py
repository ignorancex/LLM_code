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
import random

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

def sample_docs(doc_list, num):
    res = []
    if len(doc_list) >= num:
        res = sample(doc_list, num)
    else:
        res += doc_list
        for i in range(num-len(doc_list)):
            res += sample(doc_list, 1)
    return res

def shuffle_docs(doc_list):
    res = [] # To avoid in-place modification
    res += doc_list
    random.Random(2025).shuffle(res)
    gold_answer_list = []
    for d in res:
        gold_answer_list += d['gold_answers']
    return res, [d['lang'] for d in res], list(set(gold_answer_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mname", type=str, default="CohereForAI/aya-expanse-8b", help="LLM name on HF")
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size")

    args = parser.parse_args()

    task = 'MKQA'
    num_retrieval = 4
    # Load data
    load_path = f"data_inlang_{task}_filter/{args.lang}_gtr_top30.json"
    with open(load_path, encoding='utf-8') as f:
        data = json.load(f)
    f.close()
 
    save_dir = f"results/{task}_open_multi/"

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
    # debug
    # data = data[:10]

    num_ins = len(data)
    batch_prompts = []
    save_data = []
    start_id = 0
    count_ins = 0
    for ins_id, ins in enumerate(tqdm(data)):
        try:
            # Prepare inlang gold docs and outlang gold docs
            with open(f"data_inlang_{task}_filter/golddoc/{ins['index']}.jsonl") as f:
                gold_docs = f.read()
                gold_docs = gold_docs.split('\n')[:-1]
                gold_docs = [json.loads(i) for i in gold_docs]
            f.close()
        except Exception as e:
            print(f"Error: {e}")
            gold_docs = []

        try:
            with open(f"data_inlang_{task}_filter/distractor/{ins['index']}.jsonl") as f:
                dist_docs = f.read()
                dist_docs = dist_docs.split('\n')[:-1]
                dist_docs = [json.loads(i) for i in dist_docs]
            f.close()
        except Exception as e:
            print(f"Error: {e}")
            dist_docs = []

        # filter inlang and outlang gold docs
        gold_docs_inlang_cand = []
        gold_docs_outlang_cand = []
        for gd in gold_docs:
            if gd['lang'] == args.lang:
                gold_docs_inlang_cand.append(gd)
            else:
                gold_docs_outlang_cand.append(gd)
        
        # filter inlang and outlang distractors
        dist_docs_inlang_cand = []
        dist_docs_outlang_cand = []
        for dd in dist_docs:  
            if dd['lang'] == args.lang:
                dist_docs_inlang_cand.append(dd)
            else:
                dist_docs_outlang_cand.append(dd)
        
        if gold_docs_inlang_cand and gold_docs_outlang_cand and dist_docs_inlang_cand and dist_docs_outlang_cand:
            count_ins += 1

            #doc_template = "Document [{ID}](Title: {T}): {P}\n"
            doc_template = "[{ID}]({T}): {P}\n"
            
            meta_info = {"index": ins['index'], "answer": list(set(ins['gold_answers']))}
            prompts = []

            # (1) non context
            #content_non_ctx = "Question: " + ins['question'] + " Answer:"
            content_non_ctx = f"{ins['question']}\n"
            prompt_non_ctx = make_prompt(tokenizer, args.mname, instructions[args.lang]['non_ctx'], content_non_ctx)
            prompts += [prompt_non_ctx]

            # (2) 1 outlang gold 
            for i in range(3):
                # (2.1) no distractor
                # (2.2) 3 inlang distractors
                # (2.3) 3 outlang distractors
                gold_docs_outlang = sample_docs(gold_docs_outlang_cand, 1)
                dist_docs_inlang  = sample_docs(dist_docs_inlang_cand, 3)
                dist_docs_outlang = sample_docs(dist_docs_outlang_cand, 3)

                docs_no, gold_docs_outlang_langs, gold_ans_list_out = shuffle_docs(gold_docs_outlang)
                docs_in, dist_docs_inlang_langs, _  = shuffle_docs(gold_docs_outlang+dist_docs_inlang)
                docs_out, dist_docs_outlang_langs, gold_answer_list_out_dist = shuffle_docs(gold_docs_outlang+dist_docs_outlang)

                context_no  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_no)]
                context_in  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_in)]
                context_out = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_out)]
                
                content_no  = f"{''.join(context_no)}{ins['question']}\n"
                content_in  = f"{''.join(context_in)}{ins['question']}\n"
                content_out = f"{''.join(context_out)}{ins['question']}\n"

                prompt_no  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_no)
                prompt_in  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_in)
                prompt_out = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_out)

                prompts += [prompt_no, prompt_in, prompt_out]
                meta_info[f'lang_gold_1o_{i}'] = gold_docs_outlang_langs
                meta_info[f'lang_dist_1o3i_{i}'] = dist_docs_inlang_langs
                meta_info[f'lang_dist_1o3o_{i}'] = dist_docs_outlang_langs
                meta_info[f'answer_out_1o3o_{i}'] = gold_ans_list_out
                meta_info[f'answer_out_dist_1o3o_{i}'] = [gd_ans for gd_ans in gold_answer_list_out_dist if gd_ans not in gold_ans_list_out]

            # (3) 3 outlang gold
            for i in range(3):
                # (3.1) no distractor
                # (3.2) 1 inlang distractor
                # (3.3) 1 outlang distractor
                gold_docs_outlang = sample_docs(gold_docs_outlang_cand, 3)
                dist_docs_inlang  = sample_docs(dist_docs_inlang_cand, 1)
                dist_docs_outlang = sample_docs(dist_docs_outlang_cand, 1)

                docs_no, gold_docs_outlang_langs, gold_ans_list_out = shuffle_docs(gold_docs_outlang)
                docs_in, dist_docs_inlang_langs, _   = shuffle_docs(gold_docs_outlang+dist_docs_inlang)
                docs_out, dist_docs_outlang_langs, gold_answer_list_out_dist = shuffle_docs(gold_docs_outlang+dist_docs_outlang)

                context_no  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_no)]
                context_in  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_in)]
                context_out = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_out)]
                
                content_no  = f"{''.join(context_no)}{ins['question']}\n"
                content_in  = f"{''.join(context_in)}{ins['question']}\n"
                content_out = f"{''.join(context_out)}{ins['question']}\n"

                prompt_no  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_no)
                prompt_in  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_in)
                prompt_out = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_out)

                prompts += [prompt_no, prompt_in, prompt_out]
                meta_info[f'lang_gold_3o_{i}']   = gold_docs_outlang_langs
                meta_info[f'lang_dist_3o1i_{i}'] = dist_docs_inlang_langs
                meta_info[f'lang_dist_3o1o_{i}'] = dist_docs_outlang_langs
                meta_info[f'answer_out_3o1o_{i}'] = gold_ans_list_out
                meta_info[f'answer_out_dist_3o1o_{i}'] = [gd_ans for gd_ans in gold_answer_list_out_dist if gd_ans not in gold_ans_list_out]

            # (4) 1 inlang gold 
            for i in range(3):
                # (4.1) no distractor
                # (4.2) 3 inlang distractor
                # (4.3) 3 outlang distractor
                gold_docs_inlang = sample_docs(gold_docs_inlang_cand, 1)
                dist_docs_inlang  = sample_docs(dist_docs_inlang_cand, 3)
                dist_docs_outlang = sample_docs(dist_docs_outlang_cand, 3)

                docs_no, gold_docs_inlang_langs, _ = shuffle_docs(gold_docs_inlang)
                docs_in, dist_docs_inlang_langs, _ = shuffle_docs(gold_docs_inlang+dist_docs_inlang)
                docs_out, dist_docs_outlang_langs, gold_answer_list_out_dist = shuffle_docs(gold_docs_inlang+dist_docs_outlang)

                context_no  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_no)]
                context_in  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_in)]
                context_out = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_out)]
                
                content_no  = f"{''.join(context_no)}{ins['question']}\n"
                content_in  = f"{''.join(context_in)}{ins['question']}\n"
                content_out = f"{''.join(context_out)}{ins['question']}\n"

                prompt_no  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_no)
                prompt_in  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_in)
                prompt_out = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_out)

                prompts += [prompt_no, prompt_in, prompt_out]
                meta_info[f'lang_gold_1i_{i}'] = gold_docs_inlang_langs
                meta_info[f'lang_dist_1i3i_{i}'] = dist_docs_inlang_langs
                meta_info[f'lang_dist_1i3o_{i}'] = dist_docs_outlang_langs
                gold_ans_list_out = []
                meta_info[f'answer_out_1i3o_{i}'] = gold_ans_list_out
                meta_info[f'answer_out_dist_1i3o_{i}'] = [gd_ans for gd_ans in gold_answer_list_out_dist if gd_ans not in gold_ans_list_out]

            # (5) 3 inlang gold
            for i in range(3):
                # (5.1) no distractor
                # (5.2) 1 inlang distractor
                # (5.3) 1 outlang distractor
                gold_docs_inlang = sample_docs(gold_docs_inlang_cand, 3)
                dist_docs_inlang  = sample_docs(dist_docs_inlang_cand, 1)
                dist_docs_outlang = sample_docs(dist_docs_outlang_cand, 1)

                docs_no, gold_docs_inlang_langs, _  = shuffle_docs(gold_docs_inlang)
                docs_in, dist_docs_inlang_langs, _  = shuffle_docs(gold_docs_inlang+dist_docs_inlang)
                docs_out, dist_docs_outlang_langs, gold_answer_list_out_dist = shuffle_docs(gold_docs_inlang+dist_docs_outlang)

                context_no  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_no)]
                context_in  = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_in)]
                context_out = [doc_template.replace("{T}", doc["title"]).replace("{P}", doc["text"]).replace("{ID}", str(doc_id+1)) for doc_id, doc in enumerate(docs_out)]
                
                content_no  = f"{''.join(context_no)}{ins['question']}\n"
                content_in  = f"{''.join(context_in)}{ins['question']}\n"
                content_out = f"{''.join(context_out)}{ins['question']}\n"

                prompt_no  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_no)
                prompt_in  = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_in)
                prompt_out = make_prompt(tokenizer, args.mname, instructions[args.lang]['ctx'], content_out)

                prompts += [prompt_no, prompt_in, prompt_out]
                meta_info[f'lang_gold_3i_{i}'] = gold_docs_inlang_langs
                meta_info[f'lang_dist_3i1i_{i}'] = dist_docs_inlang_langs
                meta_info[f'lang_dist_3i1o_{i}'] = dist_docs_outlang_langs
                gold_ans_list_out = []
                meta_info[f'answer_out_3i1o_{i}'] = gold_ans_list_out
                meta_info[f'answer_out_dist_3i1o_{i}'] = [gd_ans for gd_ans in gold_answer_list_out_dist if gd_ans not in gold_ans_list_out]
            
            save_data.append(meta_info)
            # Gather all prompts into a list
            num_prompts = len(prompts)
            batch_prompts += prompts
        

        if (((count_ins % args.batch_size) == 0) or ((ins_id+1) == num_ins)) and batch_prompts:
            # Generate responses
            sampling_params = SamplingParams(
                    temperature=1.0,
                    top_k=1,
                    max_tokens=100,
                )
            #prompt_logprobs=0,
            seg_raw_responses_list = vmodel.generate(batch_prompts, sampling_params, use_tqdm=True)

            mapping = {
                0: "non_ctx",
                1: "1o_0", 2: "1o3i_0", 3: "1o3o_0", 4: "1o_1", 5: "1o3i_1", 6: "1o3o_1", 7: "1o_2", 8: "1o3i_2", 9: "1o3o_2",
                10: "3o_0", 11: "3o1i_0", 12: "3o1o_0", 13: "3o_1", 14: "3o1i_1", 15: "3o1o_1", 16: "3o_2", 17: "3o1i_2", 18: "3o1o_2",
                19: "1i_0", 20: "1i3i_0", 21: "1i3o_0", 22: "1i_1", 23: "1i3i_1", 24: "1i3o_1", 25: "1i_2", 26: "1i3i_2", 27: "1i3o_2",
                28: "3i_0", 29: "3i1i_0", 30: "3i1o_0", 31: "3i_1", 32: "3i1i_1", 33: "3i1o_1", 34: "3i_2", 35: "3i1i_2", 36: "3i1o_2"
                }
            # Obtain responses for each prompt
            for raw_response_id, raw_response in enumerate(seg_raw_responses_list):
                current_id = start_id+int(raw_response_id/num_prompts)
                remainder = (raw_response_id % len(prompts))
                if remainder in mapping.keys():
                    field_prompt = f'prompt_{mapping[remainder]}'
                    field_response = f'response_{mapping[remainder]}'
                    field_correct = f'correct_{mapping[remainder]}'
                    field_correct_out = f'correct_out_{mapping[remainder]}'
                    field_correct_out_dist = f'correct_out_dist_{mapping[remainder]}'
                else:
                    raise ValueError('Number of prompts does not match. Please double check.')

                save_data[current_id][field_prompt] = raw_response.prompt
                save_data[current_id][field_response] = raw_response.outputs[0].text
                save_data[current_id][field_correct] = check_substring(save_data[current_id]['answer'], raw_response.outputs[0].text)
                if remainder in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]:
                    save_data[current_id][field_correct_out] = check_substring(save_data[current_id][f'answer_out_{mapping[remainder]}'], raw_response.outputs[0].text)
                    save_data[current_id][field_correct_out_dist] = check_substring(save_data[current_id][f'answer_out_dist_{mapping[remainder]}'], raw_response.outputs[0].text)

            batch_prompts = []
            start_id = start_id + int(len(seg_raw_responses_list)/num_prompts)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}{args.mname.split('/')[-1]}_{args.lang}.json", 'w', encoding='utf-8') as f:
        # json.dump(save_data, f, indent=4, sort_keys=False)
        json.dump(save_data, f, sort_keys=False)
    print(len(save_data))

if __name__ == "__main__":
    main()
