import json
from tqdm import tqdm
import os
import sys

from collections import defaultdict
import gc
import psutil

langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']

langs = langs_pro + langs_com + langs_mtr

save_file = "has_answer_list_GMMLU.txt"
if not os.path.exists(save_file):
    #langs = langs_pro
    #langs = langs_com
    #langs = langs_mtr

    has_gold_set = set()
    for lang in tqdm(langs):
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"RSS: {memory_info.rss / 1024**2} MB")
        print(f"VMS: {memory_info.vms / 1024**2} MB")

        with open(f"data_inlang_GMMLU/{lang}_gtr_top30.json") as f:
            data = json.load(f)
        f.close()
        for index, ins in enumerate(data):
            if ins['subject'] == "abstract_algebra": continue
            if ins['has_gold']:
                has_gold_set.add(index)
        del data
        gc.collect()

    has_answer_list = sorted(list(has_gold_set))
    with open(save_file, 'w') as f:
        f.write(str(has_answer_list))
else:
    with open(save_file) as f:
        has_answer_list = eval(f.read())
    f.close()

print(has_answer_list)
print(len(has_answer_list))

for lang in langs:
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024**2} MB")
    print(f"VMS: {memory_info.vms / 1024**2} MB")

    print(lang)
    save_data = []
    with open(f"data_inlang_GMMLU/{lang}_gtr_top30.json") as f:
        data = json.load(f)
    f.close()
    for index, ins in enumerate(tqdm(data)):
        if index not in has_answer_list: continue
        if ins["subject"] == "abstract_algebra": continue
        tmp_ins = dict()
        tmp_ins["index"] = index
        for k, v in ins.items():
            tmp_ins[k] = v
        save_data.append(tmp_ins)

    print(len(save_data))
    print("=======")
    
    if not os.path.exists("data_inlang_GMMLU_filter/"):
        os.makedirs("data_inlang_GMMLU_filter/")

    with open(f"data_inlang_GMMLU_filter/{lang}_gtr_top30.json", 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, sort_keys=False)
    del data
    del save_data
    del tmp_ins
    gc.collect()
