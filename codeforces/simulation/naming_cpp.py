import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict

# === 1. 命名模式定义 ===
naming_patterns = {
    'single_letter': r'^[a-zA-Z]$',
    'lowercase': r'^[a-z]+$',
    'UPPERCASE': r'^[A-Z]+$',
    'camelCase': r'^[a-z]+(?:[A-Z][a-z]*)*$',
    'snake_case': r'^[a-z]+(?:_[a-z]+)+$',
    'PascalCase': r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$',
    'endsWithDigits': r'^[A-Za-z_]+[0-9]+$',
    'Other': r'.*'
}
compiled_patterns = {k: re.compile(v) for k, v in naming_patterns.items()}

def get_naming_category(name: str) -> str:
    for key, pat in compiled_patterns.items():
        if pat.fullmatch(name):
            return key
    return 'Other'

# === 2. C++ 分析函数，返回 counters + 名称长度总和与计数 ===
func_def_re = re.compile(
    r'\b([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)\s*\([^;]*\)\s*\{|'      # 类成员函数 A::foo(
    r'\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{'  # 普通函数 int foo(
)
var_decl_re = re.compile(
    r'\b(?:unsigned\s+)?(?:int|long|short|float|double|char|bool|auto|std::\w+<[^>]+>)\s+([A-Za-z_]\w*)'
)

def analyze_cpp_code(code: str):
    func_counter = Counter()
    var_counter = Counter()
    sum_func_len = 0
    count_func = 0
    sum_var_len = 0
    count_var = 0

    # 去除注释
    code_nc = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.DOTALL | re.MULTILINE)

    # 找函数名
    for m in func_def_re.finditer(code_nc):
        name = m.group(2) or m.group(4) or m.group(1)
        if name:
            cat = get_naming_category(name)
            func_counter[cat] += 1
            sum_func_len += len(name)
            count_func += 1

    # 找变量名
    for m in var_decl_re.finditer(code_nc):
        name = m.group(1)
        cat = get_naming_category(name)
        var_counter[cat] += 1
        sum_var_len += len(name)
        count_var += 1

    return var_counter, func_counter, sum_var_len, count_var, sum_func_len, count_func

# === 3. 读取输入 JSON 并统计 ===
input_files = {
    'DeepSeek': 'LLM_code/codeforces/simulation/valid/DeepSeek_cpp.json',
    'Gemma':    'LLM_code/codeforces/simulation/valid/Gemma_cpp.json',
    'Qwen':     'LLM_code/codeforces/simulation/valid/Qwen_cpp.json',
    'Gemini':   'LLM_code/codeforces/simulation/valid/Gemini_cpp.json',
    'GPT':      'LLM_code/codeforces/simulation/valid/GPT_cpp.json',
    'Llama':    'LLM_code/codeforces/simulation/valid/Llama4_cpp.json',
}

variable_result = {'cpp': defaultdict(dict)}
function_result = {'cpp': defaultdict(dict)}
field_map = {
    'sourceCode': 'ac',
    'generate_code_block': 'ans',
    'generate_ref_code_block': 'ref'
}

for model_name, file_path in input_files.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    for field, label in field_map.items():
        total_var = 0
        total_func = 0
        var_counter = Counter()
        func_counter = Counter()
        sum_var_len = 0
        sum_func_len = 0
        cnt_var = 0
        cnt_func = 0

        for item in items:
            code = item.get(field, '')
            v_cnt, f_cnt, v_len, v_ct, f_len, f_ct = analyze_cpp_code(code)
            var_counter.update(v_cnt)
            func_counter.update(f_cnt)
            total_var += sum(v_cnt.values())
            total_func += sum(f_cnt.values())
            sum_var_len += v_len
            cnt_var += v_ct
            sum_func_len += f_len
            cnt_func += f_ct

        # 计算比例并加入平均长度
        variable_result['cpp'][model_name][label] = {
            **{k: (var_counter[k] / total_var if total_var else 0.0) for k in naming_patterns},
            'avg_length': (sum_var_len / cnt_var if cnt_var else 0.0)
        }
        function_result['cpp'][model_name][label] = {
            **{k: (func_counter[k] / total_func if total_func else 0.0) for k in naming_patterns},
            'avg_length': (sum_func_len / cnt_func if cnt_func else 0.0)
        }

# === 4. 写入结果 JSON ===
os.makedirs('LLM_code/codeforces/simulation/result', exist_ok=True)
with open('LLM_code/codeforces/simulation/result/variable_naming_all_models_cpp.json', 'w', encoding='utf-8') as f:
    json.dump(variable_result, f, indent=2, ensure_ascii=False)
with open('LLM_code/codeforces/simulation/result/function_naming_all_models_cpp.json', 'w', encoding='utf-8') as f:
    json.dump(function_result, f, indent=2, ensure_ascii=False)

print("✅ Finished writing C++ naming and length stats.")
