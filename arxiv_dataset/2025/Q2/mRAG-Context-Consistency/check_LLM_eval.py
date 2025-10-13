import os
import json
import argparse
from collections import defaultdict

langs = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mname", type=str, default="CohereForAI/aya-expanse-8b", help="LLM name on HF")
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument("--dataset", type=str, default="XQUAD_open", help="Dataset")
    args = parser.parse_args()

    with open(f'LLM_eval/sample_results_{args.dataset}_{args.mname.split("/")[-1]}_{args.lang}.json', 'r') as f:
        data = json.load(f)
    f.close()
    
    original_res = {k: 0 for k in langs}
    original_res_out = {k: 0 for k in langs}

    res = {k: 0 for k in langs}
    res_out = {k: 0 for k in langs}
    reason = {k: defaultdict(int) for k in langs}
    for d in data:
        LLM_eval = d['LLM_eval']
        for lang in langs:
            d_eval = json.loads(LLM_eval[lang].replace('```', '').replace('json', ''))
            if d_eval['correct']: res[lang] += 1
            if not d_eval['correct'] and d_eval['correct_passage']: res_out[lang] += 1
            if not d_eval['correct'] and not d_eval['correct_passage']: reason[lang][d_eval['wrong_reason']] += 1
            if d[f'correct_ctx_{lang}']: original_res[lang] += 1
            if not d[f'correct_ctx_{lang}'] and d[f'correct_ctx_{lang}_out']: original_res_out[lang] += 1

    print(args.lang)
    print(original_res)
    print(original_res_out)
    print(res)
    print(res_out)
    res_reason = dict()
    for k, v in reason.items():
        if k == args.lang: continue
        res_reason[k] = dict(v)
    print(res_reason)
    print()
    print((sum(original_res.values())-res[args.lang])/11)
    print((sum(original_res_out.values())-res_out[args.lang])/11)

    print((sum(res.values())-res[args.lang])/11)
    print((sum(res_out.values())-res_out[args.lang])/11)

if __name__ == '__main__':
    main()