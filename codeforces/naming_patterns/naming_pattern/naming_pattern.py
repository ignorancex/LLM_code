import pandas as pd
import re
import json
import os
from collections import defaultdict
naming_patterns = {'single_letter': '^[a-zA-Z]$', 'lowercase': '^[a-z]+$', 'UPPERCASE': '^[A-Z]+$', 'camelCase': '^[a-z]+(?:[A-Z][a-z]*)*$', 'snake_case': '^[a-z]+(?:_[a-z]+)+$', 'PascalCase': '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', 'UPPER_SNAKE_CASE': '^[A-Z]+(?:_[A-Z]+)+$', 'endsWithDigits': '^[A-Za-z_]+[0-9]+$', 'Other': '.*'}

def classify_name(name):
    for (pattern_name, pattern) in naming_patterns.items():
        if re.match(pattern, name):
            return pattern_name
    return 'Other'
raw_result_funcs = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(int))))
raw_result_vars = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(int))))
folder_path = 'LLM_code/codeforces/result'
files = [f for f in os.listdir(folder_path) if f.endswith('_funcs.csv') or f.endswith('_vars.csv')]
for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    if 'cpp' in file:
        language = 'cpp'
    elif 'python' in file:
        language = 'python'
    else:
        language = 'unknown'
    if 'deepseek_32b' in file:
        model = 'deepseek_32b'
    elif 'gemma_27b' in file:
        model = 'gemma_27b'
    elif 'qwen_32b' in file:
        model = 'qwen_32b'
    else:
        model = 'unknown'
    if '_funcs.csv' in file:
        current_result = raw_result_funcs
    elif '_vars.csv' in file:
        current_result = raw_result_vars
    else:
        continue
    for (_, row) in df.iterrows():
        name = row['name']
        ac_count = row['ac_count']
        ans_count = row['ans_count']
        ref_count = row['ref_count']
        pattern = classify_name(name)
        current_result[language][model]['ac'][pattern] += ac_count
        current_result[language][model]['ans'][pattern] += ans_count
        current_result[language][model]['ref'][pattern] += ref_count

def normalize_result(raw_result):
    final_result = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
    for language in raw_result:
        for model in raw_result[language]:
            for category in ['ac', 'ans', 'ref']:
                total = sum(raw_result[language][model][category].values())
                for pattern in raw_result[language][model][category]:
                    if total > 0:
                        final_result[language][model][category][pattern] = raw_result[language][model][category][pattern] / total
                    else:
                        final_result[language][model][category][pattern] = 0.0
    return final_result
final_result_funcs = normalize_result(raw_result_funcs)
final_result_vars = normalize_result(raw_result_vars)
os.makedirs('LLM_code/codeforces/naming_pattern', exist_ok=True)
funcs_output_path = 'LLM_code/codeforces/naming_pattern/naming_pattern_distribution_funcs.json'
vars_output_path = 'LLM_code/codeforces/naming_pattern/naming_pattern_distribution_vars.json'
with open(funcs_output_path, 'w', encoding='utf-8') as f:
    json.dump(final_result_funcs, f, ensure_ascii=False, indent=4)
with open(vars_output_path, 'w', encoding='utf-8') as f:
    json.dump(final_result_vars, f, ensure_ascii=False, indent=4)