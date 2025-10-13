import json
import random

mname = 'aya-expanse-8b_{}.json'
langs = ['zh', 'es']

all_langs = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh']

for lang in langs:
    fname = f'results/XQUAD_open/{mname.format(lang)}'
    data = json.load(open(fname))
    data = [d for d in data if d[f'correct_ctx_{lang}'] and not d['correct_non_ctx'] and not d['correct_non_ctx_out']]
    filtered_data = []
    for d in data:
        correct_outlang = [l for l in all_langs if d[f'correct_ctx_{l}']]
        correct_outlang_out = [l for l in all_langs if d[f'correct_ctx_{l}_out']]
        
        if (len(correct_outlang) < 2) and (len(correct_outlang_out) < 2) and d['index'] != 173:
            filtered_data.append(d)
    print(len(filtered_data))
    random.seed(123)
    sample_data = random.sample(filtered_data, 20)

    with open(f'error_analysis_{lang}.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4, ensure_ascii=False)
