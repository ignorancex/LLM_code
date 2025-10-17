import json
import re
import ast
from pathlib import Path
from collections import Counter, defaultdict
import os

naming_patterns = {
    'single_letter': '^[a-zA-Z]$',
    'lowercase':     '^[a-z]+$',
    'UPPERCASE':     '^[A-Z]+$',
    'camelCase':     '^[a-z]+(?:[A-Z][a-z]*)*$',
    'snake_case':    '^[a-z]+(?:_[a-z]+)+$',
    'PascalCase':    '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$',
    'endsWithDigits':'^[A-Za-z_]+[0-9]+$',
    'Other':         '.*'
}
compiled_patterns = {k: re.compile(v) for k, v in naming_patterns.items()}

def get_naming_category(name: str) -> str:
    for key, pat in compiled_patterns.items():
        if pat.fullmatch(name):
            return key
    return 'Other'

def analyze_code(code: str):
    var_counter = Counter()
    func_counter = Counter()
    sum_var_len = 0
    sum_func_len = 0
    count_var = 0
    count_func = 0

    try:
        tree = ast.parse(code)
    except Exception:
        return var_counter, func_counter, sum_var_len, count_var, sum_func_len, count_func

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            cat = get_naming_category(name)
            var_counter[cat] += 1
            sum_var_len += len(name)
            count_var += 1
        elif isinstance(node, ast.FunctionDef):
            name = node.name
            cat = get_naming_category(name)
            func_counter[cat] += 1
            sum_func_len += len(name)
            count_func += 1

    return var_counter, func_counter, sum_var_len, count_var, sum_func_len, count_func

unique_path = Path('dataset/unique_problem_python.json')
with unique_path.open('r', encoding='utf-8') as f:
    unique_items = json.load(f)
ac_map = {item['submission_id']: item['sourceCode'] for item in unique_items}

input_files = {
    'DeepSeek': 'LLM_code/codeforces/simulation/output/DeepSeek_python.json',
    'Gemma':    'LLM_code/codeforces/simulation/output/Gemma_python.json',
    'Qwen':     'LLM_code/codeforces/simulation/output/Qwen_python.json',
    'Gemini':   'LLM_code/codeforces/simulation/output/Gemini_python.json',
    'GPT':      'LLM_code/codeforces/simulation/output/GPT_python.json',
    'Llama':    'LLM_code/codeforces/simulation/output/Llama4_python.json',
}

variable_result = {'python': defaultdict(dict)}
function_result = {'python': defaultdict(dict)}
field_map = {
    'sourceCode':            'ac',    # human code, now from unique map
    'generate_code_block':   'ans',
    'generate_ref_code_block':'ref'
}

for model_name, file_path in input_files.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    for field, label in field_map.items():
        # counters and sums
        total_var_counts = 0
        total_func_counts = 0
        var_counter = Counter()
        func_counter = Counter()
        sum_var_len = 0
        sum_func_len = 0
        count_var = 0
        count_func = 0

        for item in items:
            if label == 'ac':
                sid = item.get('submission_id')
                code = ac_map.get(sid, '')
            else:
                code = item.get(field, '')

            v_cnt, f_cnt, svl, cvar, sfl, cfunc = analyze_code(code)
            var_counter.update(v_cnt)
            func_counter.update(f_cnt)
            total_var_counts  += sum(v_cnt.values())
            total_func_counts += sum(f_cnt.values())
            sum_var_len  += svl
            count_var    += cvar
            sum_func_len += sfl
            count_func   += cfunc

        variable_result['python'][model_name][label] = {
            **{k: (var_counter[k] / total_var_counts if total_var_counts else 0.0)
               for k in naming_patterns},
            'avg_length': (sum_var_len / count_var if count_var else 0.0)
        }
        function_result['python'][model_name][label] = {
            **{k: (func_counter[k] / total_func_counts if total_func_counts else 0.0)
               for k in naming_patterns},
            'avg_length': (sum_func_len / count_func if count_func else 0.0)
        }

os.makedirs('LLM_code/codeforces/simulation/result', exist_ok=True)
with open('LLM_code/codeforces/simulation/result/variable_naming_all_models_python.json', 'w', encoding='utf-8') as f:
    json.dump(variable_result, f, indent=2, ensure_ascii=False)
with open('LLM_code/codeforces/simulation/result/function_naming_all_models_python.json', 'w', encoding='utf-8') as f:
    json.dump(function_result, f, indent=2, ensure_ascii=False)

print("✅ Completed Python naming & length stats with shared AC.")
