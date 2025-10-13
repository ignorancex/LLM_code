import json
import gc
from tqdm import tqdm
from collections import defaultdict
import psutil

langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']

#langs = langs_pro + langs_com + langs_mtr
langs = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'ko', 'km', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh']

dataset = 'MKQA'

with open(f"has_answer_list_{dataset}.txt") as f:
    has_answer_list = eval(f.read())
f.close()

num_row = 14

res = dict()

for lang_id, lang in enumerate(langs):
    gold_in = 0
    gold_out = 0
    gold_both = 0
    for index in tqdm(has_answer_list):
        with open(f'data_inlang_{dataset}_filter/golddoc/{index}.jsonl', encoding='utf-8') as f:
            docs = f.read().split('\n')[:-1]
            docs = [json.loads(i) for i in docs]
        f.close()

        gold_in_case = False
        gold_out_case = False
        for doc in docs:
            if doc['lang'] == lang:
                gold_in_case = True
            if doc['lang'] != lang:
                gold_out_case = True
        gold_in += gold_in_case
        gold_out += gold_out_case
        gold_both += (gold_in_case and gold_out_case)

        del docs
        gc.collect()
    
    res[lang] = dict()
    res[lang]['in'] = str(gold_in)
    res[lang]['out'] = str(gold_out)
    res[lang]['both'] = str(gold_both)

sorted_langs = list(zip(*sorted(res.items(), key=lambda item: int(item[1]['in']), reverse=True)))[0]

l0 = 'Query Lang.'
l1 = '\\# Inlang'
l2 = '\\# Outlang'
l3 = '\\# Both'
for lang_id, lang in enumerate(sorted_langs):
    l0 += f' & {lang}'
    l1 += f' & {res[lang]['in']}'
    l2 += f' & {res[lang]['out']}'
    l3 += f' & {res[lang]['both']}'
    
    if (lang_id + 1) % num_row == 0 or (lang_id +1) == len(langs):
        print(l0 + " \\\\")
        print('\\midrule')
        print(l1 + " \\\\")
        print(l2 + " \\\\")
        print(l3 + " \\\\")
        print('\\midrule')
        l0 = 'Query Lang.'
        l1 = '\\# Inlang'
        l2 = '\\# Outlang'
        l3 = '\\# Both'
