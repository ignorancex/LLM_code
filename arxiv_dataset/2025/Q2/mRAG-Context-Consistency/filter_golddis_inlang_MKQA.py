import json
from tqdm import tqdm
import os
import sys

from collections import defaultdict
import gc
import psutil
import jsonlines

#langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
#langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
#langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']
#langs = langs_pro + langs_com + langs_mtr

langs = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'ko', 'km', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh']

with open("has_answer_list_MKQA.txt") as f:
    has_answer_list = eval(f.read())
f.close()
    
gold_docs = defaultdict(list)
distractors = defaultdict(list)

if not os.path.exists("data_inlang_MKQA_filter/golddoc/"):
    os.makedirs("data_inlang_MKQA_filter/golddoc/")

if not os.path.exists("data_inlang_MKQA_filter/distractor/"):
    os.makedirs("data_inlang_MKQA_filter/distractor/")


for lang in langs:
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024**2} MB")
    print(f"VMS: {memory_info.vms / 1024**2} MB")
    print(lang)
    with open(f"data_inlang_MKQA/{lang}_gtr_top30.json") as f:
        data = json.load(f)
    f.close()
    for index, ins in enumerate(tqdm(data)):
        if index not in has_answer_list: continue
        for doc in ins['docs']:
            tmp_doc = dict()
            for k, v in doc.items():
                tmp_doc['lang'] = lang
                tmp_doc[k] = v
                
            if tmp_doc['has_answer_text'] or tmp_doc['has_answer_title']: 
                with open(f'data_inlang_MKQA_filter/golddoc/{index}.jsonl', mode='a') as f:
                    json.dump(tmp_doc, f)
                    f.write('\n')
                f.close()
            else:
                with open(f'data_inlang_MKQA_filter/distractor/{index}.jsonl', mode='a') as f:
                    json.dump(tmp_doc, f)
                    f.write('\n')
                f.close()
    del data
    del tmp_doc
    gc.collect()
