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

langs = ['en', 'ar', 'zh', 'si', 'yo']

if not os.path.exists("has_answer_list_GMMLU.txt"):
    raise ValueError('Please run filter_subset_inlang_GMMLU.py firstly to get the overlapping subset index file.')
else:
    with open("has_answer_list_GMMLU.txt") as f:
        has_answer_list = eval(f.read())
    f.close()
    
for lang in langs:
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024**2} MB")
    print(f"VMS: {memory_info.vms / 1024**2} MB")

    print(lang)
    save_data = []
    with open(f"data_cross_GMMLU/{lang}_gtr_top30.json") as f:
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
    
    if not os.path.exists("data_cross_GMMLU_filter/"):
        os.makedirs("data_cross_GMMLU_filter/")

    with open(f"data_cross_GMMLU_filter/{lang}_gtr_top30.json", 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, sort_keys=False)
    del data
    del save_data
    del tmp_ins
    gc.collect()
