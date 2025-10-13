import json
from tqdm import tqdm
import os
import sys

from collections import defaultdict
import gc
import psutil
import jsonlines

langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']

langs = langs_pro + langs_com + langs_mtr
for lang in tqdm(langs):
    with open(f"data_inlang_GMMLU_raw/{lang}_gtr_top30.json") as f:
        data = json.load(f)
    f.close()

    save_data = []
    for ins_id, ins in enumerate(tqdm(data)):
        tmp_dict = dict()
        tmp_dict['id'] = ins['id']
        tmp_dict['subject'] = ins['subject']
        tmp_dict['subject_category'] = ins['subject_category']
        tmp_dict['question'] = ins['question']
        tmp_dict['A'] = ins['A']
        tmp_dict['B'] = ins['B']
        tmp_dict['C'] = ins['C']
        tmp_dict['D'] = ins['D']
        tmp_dict['answer'] = ins['answer']
        # The index of the missing-value questions will be outputted.
        # We manually translate the value from English to fix the NULL values.
        if not (tmp_dict['A'] and tmp_dict['B'] and tmp_dict['C'] and tmp_dict['D'] and tmp_dict['answer']):
            print(lang)
            print(ins_id)
            print(tmp_dict['id'])
            print()
        tmp_dict['answer_text'] = ins['answer_text']
        tmp_dict['has_gold'] = False
        tmp_dict['docs'] = []
        for doc in ins['docs']:
            tmp_doc = dict()
            tmp_doc['title'] = doc['title']
            tmp_doc['text'] = doc['text']
            tmp_doc['score'] = doc['score']
            tmp_doc['has_answer_title'] = (tmp_dict['answer_text'].lower() in tmp_doc['title'].lower())
            tmp_doc['has_answer_text'] = (tmp_dict['answer_text'].lower() in tmp_doc['text'].lower())
            tmp_doc['url'] = doc['url']
            tmp_dict['has_gold'] = (tmp_dict['has_gold'] or tmp_doc['has_answer_title'] or tmp_doc['has_answer_text'])
            tmp_dict['docs'].append(tmp_doc)
        save_data.append(tmp_dict)

        del tmp_doc
        del tmp_dict
    del data    
    gc.collect()

    with open(f"data_inlang_GMMLU/{lang}_gtr_top30.json", 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, sort_keys=False)


