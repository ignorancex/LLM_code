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

langs = langs_pro + langs_com + langs_mtr
#langs = ['en', 'ar', 'zh', 'si', 'yo']

for lang in langs:
    print(lang)
    with open(f"data_cross_GMMLU_raw/{lang}_gtr_top30.json") as f:
        data = json.load(f)
    f.close()

    with open(f"data_inlang_GMMLU/{lang}_gtr_top30.json") as f:
        source = json.load(f)
    f.close()

    save_data = []
    for ins_id, ins in enumerate(data):
        tmp_dict = dict()
        for k, v in ins.items():
            if k in ['A', 'B', 'C', 'D', 'answer'] and not v:
                assert ins['id'] == source[ins_id]['id']
                print(ins['id'])
                tmp_dict[k] = source[ins_id][k]
            else:
                tmp_dict[k] = v

        save_data.append(tmp_dict)

    del data
    del source
    gc.collect()

    with open(f"data_cross_GMMLU/{lang}_gtr_top30.json", 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, sort_keys=False)


