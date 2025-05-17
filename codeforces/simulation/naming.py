import json
import re
import ast
from collections import Counter
import pandas as pd

# 命名风格正则
naming_patterns = {
    "single_letter": r'^[a-zA-Z]$',
    "lowercase": r'^[a-z]+$',
    "UPPERCASE": r'^[A-Z]+$',
    "camelCase": r'^[a-z]+(?:[A-Z][a-z]*)*$',
    "snake_case": r'^[a-z]+(?:_[a-z]+)+$',
    "PascalCase": r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$',
    "UPPER_SNAKE_CASE": r'^[A-Z]+(?:_[A-Z]+)+$',
    "endsWithDigits": r'^[A-Za-z_]+[0-9]+$',
    "Other": r'.*'
}
compiled_patterns = {k: re.compile(v) for k, v in naming_patterns.items()}

def get_naming_category(name):
    for key, pattern in compiled_patterns.items():
        if pattern.fullmatch(name):
            return key
    return "Other"

# 分开统计变量名与函数名
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
    return var_counter, func_counter

# 加载数据
with open("LLM_code/codeforces/simulation/deepseek_32b_python_extract_valid.json", "r", encoding="utf-8") as f:
    data = json.load(f)

fields = ["sourceCode", "generate_code_block", "generate_ref_code_block"]

# 初始化
var_stats = {field: Counter() for field in fields}
func_stats = {field: Counter() for field in fields}
var_totals = {field: 0 for field in fields}
func_totals = {field: 0 for field in fields}

# 遍历数据
for item in data:
    for field in fields:
        code = item.get(field, "").strip()
        if code:
            var_c, func_c = analyze_code_split(code)
            var_stats[field].update(var_c)
            func_stats[field].update(func_c)
            var_totals[field] += sum(var_c.values())
            func_totals[field] += sum(func_c.values())

# 构建变量命名风格比例 DataFrame
var_df = pd.DataFrame()
for field in fields:
    total = var_totals[field]
    var_df[field] = {
        k: (var_stats[field][k] / total if total > 0 else 0.0)
        for k in naming_patterns
    }
var_df = var_df.reindex(list(naming_patterns.keys()))

# 构建函数命名风格比例 DataFrame
func_df = pd.DataFrame()
for field in fields:
    total = func_totals[field]
    func_df[field] = {
        k: (func_stats[field][k] / total if total > 0 else 0.0)
        for k in naming_patterns
    }
func_df = func_df.reindex(list(naming_patterns.keys()))

# 保存（可选）
var_df.to_csv("LLM_code/codeforces/simulation/naming_ratios_variables.csv")
func_df.to_csv("LLM_code/codeforces/simulation/naming_ratios_functions.csv")

# 显示两个表格
print("=== Variable Naming Pattern Ratios ===")
print(var_df)
print("\n=== Function Naming Pattern Ratios ===")
print(func_df)
