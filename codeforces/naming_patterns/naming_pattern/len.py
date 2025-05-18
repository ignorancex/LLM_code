import os
import pandas as pd
import json
from collections import defaultdict
length_stats_funcs = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : [0, 0])))
length_stats_vars = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : [0, 0])))
folder_path = 'LLM_code/codeforces/result'
files = [f for f in os.listdir(folder_path) if f.endswith('_funcs.csv') or f.endswith('_vars.csv')]
for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    language = 'cpp' if 'cpp' in file else 'python' if 'python' in file else 'unknown'
    if 'deepseek_32b' in file:
        model = 'deepseek_32b'
    elif 'gemma_27b' in file:
        model = 'gemma_27b'
    elif 'qwen_32b' in file:
        model = 'qwen_32b'
    else:
        model = 'unknown'
    current_stats = length_stats_funcs if '_funcs.csv' in file else length_stats_vars
    for (_, row) in df.iterrows():
        name = row['name']
        name_len = len(name)
        for category in ['ac', 'ans', 'ref']:
            count = row[f'{category}_count']
            current_stats[language][model][category][0] += name_len * count
            current_stats[language][model][category][1] += count

def compute_avg_lengths(stats):
    result = defaultdict(lambda : defaultdict(dict))
    for lang in stats:
        for model in stats[lang]:
            for cat in stats[lang][model]:
                (total_len, total_count) = stats[lang][model][cat]
                avg = total_len / total_count if total_count > 0 else 0.0
                result[lang][model][cat] = round(avg, 4)
    return result
avg_funcs = compute_avg_lengths(length_stats_funcs)
avg_vars = compute_avg_lengths(length_stats_vars)
os.makedirs('LLM_code/codeforces/name_length', exist_ok=True)
funcs_path = 'LLM_code/codeforces/name_length/avg_name_length_funcs.json'
vars_path = 'LLM_code/codeforces/name_length/avg_name_length_vars.json'
with open(funcs_path, 'w', encoding='utf-8') as f:
    json.dump(avg_funcs, f, ensure_ascii=False, indent=4)
with open(vars_path, 'w', encoding='utf-8') as f:
    json.dump(avg_vars, f, ensure_ascii=False, indent=4)