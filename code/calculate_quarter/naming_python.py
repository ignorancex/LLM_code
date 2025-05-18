import os
import ast
import re
import json
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
classify_enabled = False
naming_patterns = {'single_letter': '^[a-zA-Z]$', 'lowercase': '^[a-z]+$', 'UPPERCASE': '^[A-Z]+$', 'camelCase': '^[a-z]+(?:[A-Z][a-z]*)*$', 'snake_case': '^[a-z]+(?:_[a-z]+)+$', 'PascalCase': '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', 'UPPER_SNAKE_CASE': '^[A-Z]+(?:_[A-Z]+)+$', 'endsWithDigits': '^[A-Za-z_]+[0-9]+$', 'Other': '.*'}

def get_naming_pattern(name):
    name = str(name)
    for (pattern, regex) in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return 'Other'

def extract_code_info(file_path, skipped_files_log):
    """解析 Python 代码，提取函数名、变量名"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        if '\x00' in code:
            with open(skipped_files_log, 'a', encoding='utf-8') as log:
                log.write(f'Skipped {file_path}: Contains null bytes\n')
            return (set(), set())
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        with open(skipped_files_log, 'a', encoding='utf-8') as log:
            log.write(f'Skipped {file_path}: {str(e)}\n')
        return (set(), set())
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
    return (function_names, variable_names)

def classify_category(cat):
    if cat.startswith('cs.'):
        return 'cs'
    else:
        return 'non_cs'

def process_project(project_name, quarter_path, quarter_key, quarter_repo_category, skipped_files_log):
    """处理单个项目，返回项目类别和命名模式比例"""
    project_path = os.path.join(quarter_path, project_name)
    if classify_enabled:
        project_category = quarter_repo_category.get(quarter_key, {}).get(project_name)
        if project_category is None:
            return None
    else:
        project_category = 'all'
    func_counts = defaultdict(int)
    var_counts = defaultdict(int)
    for (root, _, files) in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                (functions, variables) = extract_code_info(file_path, skipped_files_log)
                for name in functions:
                    pattern = get_naming_pattern(name)
                    func_counts[pattern] += 1
                for name in variables:
                    pattern = get_naming_pattern(name)
                    var_counts[pattern] += 1
    func_total = sum(func_counts.values())
    var_total = sum(var_counts.values())
    func_ratios = {pat: func_counts.get(pat, 0) / func_total if func_total else 0.0 for pat in naming_patterns}
    var_ratios = {pat: var_counts.get(pat, 0) / var_total if var_total else 0.0 for pat in naming_patterns}
    return (project_category, func_ratios, var_ratios)
method = 'llama_improve'
base_dir = f'LLM_code/arxiv_dataset/{method}'
output_dir = 'LLM_code/arxiv_result/naming_patterns_python'
categories_file = 'LLM_code/code/github_links/python_dataset_links_1.json'
os.makedirs(output_dir, exist_ok=True)
skipped_files_log = os.path.join(output_dir, f'skipped_files_{method}.txt')
if os.path.exists(skipped_files_log):
    os.remove(skipped_files_log)
if classify_enabled:
    with open(categories_file, 'r', encoding='utf-8') as f:
        all_categories = json.load(f)
    quarter_repo_category = defaultdict(dict)
    for (quarter, items) in all_categories.items():
        for item in items:
            link = item['link']
            categories = item['categories']
            repo_name = link.rstrip('/').split('/')[-1]
            quarter_repo_category[quarter][repo_name] = classify_category(categories)
else:
    quarter_repo_category = {}
if classify_enabled:
    category_list = ['cs', 'non_cs']
else:
    category_list = ['all']
quarter_func_ratios = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
quarter_var_ratios = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
for year in range(2020, 2026):
    max_q = 1 if year == 2025 else 4
    for q in range(1, max_q + 1):
        quarter_name = f'Q{q}'
        quarter_key = f'{year}Q{q}'
        quarter_path = os.path.join(base_dir, str(year), quarter_name)
        if not os.path.isdir(quarter_path):
            continue
        projects = [d for d in os.listdir(quarter_path) if os.path.isdir(os.path.join(quarter_path, d))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_project, proj, quarter_path, quarter_key, quarter_repo_category, skipped_files_log) for proj in projects]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f'Scanning {quarter_key}'):
                res = future.result()
                if res is None:
                    continue
                (cat, func_r, var_r) = res
                for (pat, rt) in func_r.items():
                    quarter_func_ratios[quarter_key][cat][pat].append(rt)
                for (pat, rt) in var_r.items():
                    quarter_var_ratios[quarter_key][cat][pat].append(rt)

def aggregate(ratios):
    out = {}
    for quarter in sorted(ratios.keys()):
        out[quarter] = {}
        for cat in category_list:
            out[quarter][cat] = {}
            for pat in naming_patterns:
                lst = ratios[quarter][cat][pat]
                out[quarter][cat][pat] = round(sum(lst) / len(lst), 6) if lst else 0.0
    return out
final_func = aggregate(quarter_func_ratios)
final_var = aggregate(quarter_var_ratios)
func_out = os.path.join(output_dir, f'{method}_naming_patterns_function.json')
var_out = os.path.join(output_dir, f'{method}_naming_patterns_variable.json')
with open(func_out, 'w', encoding='utf-8') as f:
    json.dump(final_func, f, ensure_ascii=False, indent=2)
with open(var_out, 'w', encoding='utf-8') as f:
    json.dump(final_var, f, ensure_ascii=False, indent=2)