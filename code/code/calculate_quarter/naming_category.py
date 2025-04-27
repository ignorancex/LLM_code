import os
import ast
import re
import csv
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import concurrent.futures

# === 定义命名方式的正则表达式 ===
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


def get_naming_pattern(name):
    name = str(name)
    for pattern, regex in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return "Other"


def extract_code_info(file_path, skipped_files_log):
    """解析 Python 代码，提取函数名、变量名"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "\x00" in code:
            with open(skipped_files_log, "a", encoding="utf-8") as log:
                log.write(f"Skipped {file_path}: Contains null bytes\n")
            return set(), set()

        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        with open(skipped_files_log, "a", encoding="utf-8") as log:
            log.write(f"Skipped {file_path}: {str(e)}\n")
        return set(), set()

    function_names = set()
    variable_names = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            function_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    variable_names.add(target.id)

    return function_names, variable_names


def classify_category(cat):
    if cat.startswith("cs."):
        if cat == "cs.LG":
            return "cs.LG"
        elif cat == "cs.CV":
            return "cs.CV"
        elif cat == "cs.CL":
            return "cs.CL"
        else:
            return "other_cs"
    else:
        return "non_cs"


def process_project(project_name, quarter_path, quarter_key, quarter_repo_category, skipped_files_log):
    """处理单个项目，返回局部统计"""
    project_path = os.path.join(quarter_path, project_name)
    project_category = quarter_repo_category.get(quarter_key, {}).get(project_name)

    if project_category is None:
        return None  # 跳过没有类别的项目

    local_func_counts = defaultdict(int)
    local_var_counts = defaultdict(int)

    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions, variables = extract_code_info(file_path, skipped_files_log)

                for name in functions:
                    pattern = get_naming_pattern(name)
                    local_func_counts[(project_category, pattern)] += 1
                for name in variables:
                    pattern = get_naming_pattern(name)
                    local_var_counts[(project_category, pattern)] += 1

    return local_func_counts, local_var_counts


# === 主程序 ===
base_dir = "LLM_code/arxiv_dataset"
output_dir = "LLM_code/naming_patterns_combined"
categories_file = "LLM_code/code/github_links/categories.json"
os.makedirs(output_dir, exist_ok=True)
skipped_files_log = os.path.join(output_dir, "skipped_files.txt")
if os.path.exists(skipped_files_log):
    os.remove(skipped_files_log)

# 加载类别信息
with open(categories_file, "r", encoding="utf-8") as f:
    all_categories = json.load(f)

# 创建季度到 仓库名->类别 的映射
quarter_repo_category = defaultdict(dict)
for quarter, items in all_categories.items():
    for item in items:
        link = item["link"]
        categories = item["categories"]
        repo_name = link.rstrip("/").split("/")[-1]
        quarter_repo_category[quarter][repo_name] = classify_category(categories)

# 最终统计表：{quarter: {category: {pattern: count}}}
quarter_func_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
quarter_var_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for year in range(2020, 2026):
    max_quarter = 1 if year == 2025 else 4
    for q in range(1, max_quarter + 1):
        quarter_name = f"Q{q}"
        year_str = str(year)
        quarter_key = f"{year_str}Q{q}"
        quarter_path = os.path.join(base_dir, year_str, quarter_name)

        if not os.path.isdir(quarter_path):
            continue

        print(f"\n🔍 Processing {quarter_key}...")

        project_list = [d for d in os.listdir(quarter_path) if os.path.isdir(os.path.join(quarter_path, d))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for project_name in project_list:
                futures.append(executor.submit(
                    process_project, project_name, quarter_path, quarter_key, quarter_repo_category, skipped_files_log
                ))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Scanning {quarter_key}"):
                result = future.result()
                if result is None:
                    continue
                local_func_counts, local_var_counts = result
                for (project_category, pattern), count in local_func_counts.items():
                    quarter_func_counts[quarter_key][project_category][pattern] += count
                for (project_category, pattern), count in local_var_counts.items():
                    quarter_var_counts[quarter_key][project_category][pattern] += count

        print(f"✅ Finished {quarter_key}")

# 构造输出 JSON，确保所有类别和所有命名方式都存在（即使为0）
all_categories_list = ["cs.LG", "cs.CV", "cs.CL", "other_cs", "non_cs"]

def compute_ratios(quarter_category_counts):
    result = {}
    for quarter in sorted(quarter_category_counts.keys()):
        result[quarter] = {}
        for cat in all_categories_list:
            pattern_counts = quarter_category_counts[quarter][cat]
            total = sum(pattern_counts.values())
            result[quarter][cat] = {}
            for pattern in naming_patterns.keys():
                if total > 0:
                    proportion = pattern_counts.get(pattern, 0) / total
                else:
                    proportion = 0.0
                result[quarter][cat][pattern] = round(proportion, 6)
    return result

final_func_output = compute_ratios(quarter_func_counts)
final_var_output = compute_ratios(quarter_var_counts)

# 保存
func_output_path = os.path.join(output_dir, "naming_patterns_function_by_category.json")
var_output_path = os.path.join(output_dir, "naming_patterns_variable_by_category.json")

with open(func_output_path, "w", encoding="utf-8") as f:
    json.dump(final_func_output, f, ensure_ascii=False, indent=2)

with open(var_output_path, "w", encoding="utf-8") as f:
    json.dump(final_var_output, f, ensure_ascii=False, indent=2)

print(f"\n🎉 All processing completed. Results saved in {func_output_path} and {var_output_path}")