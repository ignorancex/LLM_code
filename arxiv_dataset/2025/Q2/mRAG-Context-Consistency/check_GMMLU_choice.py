import re
import json
from tqdm import tqdm
from collections import defaultdict

def contains_option(option_letter, response):
    pattern = r'\b[A-D]\b'
    match = re.search(pattern, response)
    if not match:
        return False
    else:
        if option_letter == match.group(0): return True
        else: return False

def check_substring(gold_answer_list, response):
    res = False
    for ans in set(gold_answer_list):
        if ans.lower() in response.lower():
            res = True
            break
    return res

langs = ['en', 'ja', 'it', 'id', 'ko', 'nl', 'zh', 'vi', 'sv', 'pt', 'de', 'tr', 'ro', 'cs', 'ru', 'es', 'ms', 'pl', 'uk', 'fr', 'ar', 'fa', 'el', 'sr', 'he', 'hi', 'fil', 'lt', 'bn', 'ky', 'ha', 'te', 'sw', 'ig', 'si', 'ne', 'am', 'ny', 'mg', 'so', 'sn', 'yo']

res = dict()
for mname_id, mname in enumerate(["aya-expanse-8b_{}.json", "Llama-3.2-3B-Instruct_{}.json", "gemma-2-9b-it_{}.json", "Qwen2.5-7B-Instruct_{}.json"]):
    res[mname] = dict()
    print("======")
    print(mname)
    
    for lang_id, lang in enumerate(langs):
        load_path = f"results/GMMLU_choice/"+mname.format(lang)
        with open(load_path, encoding="utf-8") as f:
            data = json.load(f)
        f.close()
        
        correct_non_ctx = 0
        correct_ctx_inlang = 0
        
        correct_ctx_outlang = 0
        correct_ctx_outlang_out = 0

        for d in data:
            correct_non_ctx += d['correct_non_ctx']
            for i in range(3):
                correct_ctx_inlang += d[f'correct_ctx_inlang_{i}']
                correct_ctx_outlang += d[f'correct_ctx_outlang_{i}']

        
        length = len(data)
        correct_non_ctx = round(100*correct_non_ctx/length, 1)
        correct_ctx_inlang = round(100*correct_ctx_inlang/(3*length), 1)
        correct_ctx_outlang = round(100*correct_ctx_outlang/(3*length), 1)
        correct_ctx_outlang_out = round(100*correct_ctx_outlang_out/(3*length), 1)
        
        res[mname][lang] = {'correct_non_ctx': correct_non_ctx, 'correct_ctx_inlang': correct_ctx_inlang, 'correct_ctx_outlang':correct_ctx_outlang}

for lang in langs:
    pline = f"{lang}"
    for mname in ["aya-expanse-8b_{}.json", "Llama-3.2-3B-Instruct_{}.json", "gemma-2-9b-it_{}.json", "Qwen2.5-7B-Instruct_{}.json"]:
        pline += f"  & {res[mname][lang]['correct_non_ctx']} & {res[mname][lang]['correct_ctx_inlang']} & {res[mname][lang]['correct_ctx_outlang']}"
    print(pline + " \\\\")

print()

for mname in ["aya-expanse-8b_{}.json", "Llama-3.2-3B-Instruct_{}.json", "gemma-2-9b-it_{}.json", "Qwen2.5-7B-Instruct_{}.json"]:
    avg = [0 for _ in range(4)]
    for lang in langs:
        avg[0] += res[mname][lang]['correct_non_ctx']
        avg[1] += res[mname][lang]['correct_ctx_inlang']
        avg[2] += res[mname][lang]['correct_ctx_outlang']

    print(f"& {round(avg[0]/len(langs),1)} & {round(avg[1]/len(langs),1)} & {round(avg[2]/len(langs),1)}")

