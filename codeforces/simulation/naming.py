import json
import re
import ast
from collections import Counter
import pandas as pd
naming_patterns = {'single_letter': '^[a-zA-Z]$', 'lowercase': '^[a-z]+$', 'UPPERCASE': '^[A-Z]+$', 'camelCase': '^[a-z]+(?:[A-Z][a-z]*)*$', 'snake_case': '^[a-z]+(?:_[a-z]+)+$', 'PascalCase': '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', 'UPPER_SNAKE_CASE': '^[A-Z]+(?:_[A-Z]+)+$', 'endsWithDigits': '^[A-Za-z_]+[0-9]+$', 'Other': '.*'}
compiled_patterns = {k: re.compile(v) for (k, v) in naming_patterns.items()}

def get_naming_category(name):
    for (key, pattern) in compiled_patterns.items():
        if pattern.fullmatch(name):
            return key
    return 'Other'

def analyze_code_split(code):
    var_counter = Counter()
    func_counter = Counter()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                category = get_naming_category(node.id)
                var_counter[category] += 1
            elif isinstance(node, ast.FunctionDef):
                category = get_naming_category(node.name)
                func_counter[category] += 1
    except Exception:
        pass
    return (var_counter, func_counter)
with open('LLM_code/codeforces/simulation/deepseek_32b_python_extract_valid.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
fields = ['sourceCode', 'generate_code_block', 'generate_ref_code_block']
var_stats = {field: Counter() for field in fields}
func_stats = {field: Counter() for field in fields}
var_totals = {field: 0 for field in fields}
func_totals = {field: 0 for field in fields}
for item in data:
    for field in fields:
        code = item.get(field, '').strip()
        if code:
            (var_c, func_c) = analyze_code_split(code)
            var_stats[field].update(var_c)
            func_stats[field].update(func_c)
            var_totals[field] += sum(var_c.values())
            func_totals[field] += sum(func_c.values())
var_df = pd.DataFrame()
for field in fields:
    total = var_totals[field]
    var_df[field] = {k: var_stats[field][k] / total if total > 0 else 0.0 for k in naming_patterns}
var_df = var_df.reindex(list(naming_patterns.keys()))
func_df = pd.DataFrame()
for field in fields:
    total = func_totals[field]
    func_df[field] = {k: func_stats[field][k] / total if total > 0 else 0.0 for k in naming_patterns}
func_df = func_df.reindex(list(naming_patterns.keys()))
var_df.to_csv('LLM_code/codeforces/simulation/naming_ratios_variables.csv')
func_df.to_csv('LLM_code/codeforces/simulation/naming_ratios_functions.csv')