import json
import re
import ast
from pathlib import Path
from collections import Counter, defaultdict

# 命名风格正则定义
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

def get_naming_category(name: str) -> str:
    for key, pattern in compiled_patterns.items():
        if pattern.fullmatch(name):
            return key
    return "Other"

# 用 AST 分析变量名和函数名
def analyze_code(code: str):
    var_counter = Counter()
    func_counter = Counter()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_counter[get_naming_category(node.id)] += 1
            elif isinstance(node, ast.FunctionDef):
                func_counter[get_naming_category(node.name)] += 1
    except Exception:
        pass
    return var_counter, func_counter

# 模型名和文件路径映射
input_files = {
    "deepseek_32b": "LLM_code/codeforces/simulation/valid/deepseek_32b_python_valid.json",
    "gemma_27b": "LLM_code/codeforces/simulation/valid/gemma_27b_python_valid.json",
    "qwen_32b": "LLM_code/codeforces/simulation/valid/qwen_32b_python_valid.json",
}

# 输出结构初始化
variable_result = {"python": defaultdict(dict)}
function_result = {"python": defaultdict(dict)}

# 字段 → 标签 映射
field_map = {
    "sourceCode": "ac",
    "generate_code_block": "ans",
    "generate_ref_code_block": "ref"
}

# 主循环
for model_name, file_path in input_files.items():
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for field, label in field_map.items():
        var_total = 0
        func_total = 0
        var_counter = Counter()
        func_counter = Counter()

        for item in data:
            code = item.get(field, "")
            v_c, f_c = analyze_code(code)
            var_counter.update(v_c)
            func_counter.update(f_c)
            var_total += sum(v_c.values())
            func_total += sum(f_c.values())

        variable_result["python"][model_name][label] = {
            k: var_counter[k] / var_total if var_total else 0.0
            for k in naming_patterns
        }

        function_result["python"][model_name][label] = {
            k: func_counter[k] / func_total if func_total else 0.0
            for k in naming_patterns
        }

# 保存输出结果
with open("variable_naming_ratios_all_models.json", "w", encoding="utf-8") as f:
    json.dump(variable_result, f, indent=2, ensure_ascii=False)

with open("function_naming_ratios_all_models.json", "w", encoding="utf-8") as f:
    json.dump(function_result, f, indent=2, ensure_ascii=False)

print("✅ Done. Output saved to:")
print("- variable_naming_ratios_all_models.json")
print("- function_naming_ratios_all_models.json")
