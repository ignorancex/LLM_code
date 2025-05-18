import os
import ast
import json
from collections import defaultdict
from tqdm import tqdm
import warnings
import concurrent.futures
warnings.filterwarnings('ignore', category=SyntaxWarning)

def classify_category(cat_str):
    return 'cs' if cat_str.startswith('cs.') else 'non_cs'

def extract_code_info(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        if '\x00' in code:
            return (set(), set())
        tree = ast.parse(code)
    except Exception:
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

def process_project(project_path):
    (func_lengths, var_lengths, file_name_lengths) = ([], [], [])
    for (root, _, files) in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_name_lengths.append(len(file))
                (functions, variables) = extract_code_info(file_path)
                func_lengths.extend([len(fn) for fn in functions])
                var_lengths.extend([len(vn) for vn in variables])
    avg_func_len = sum(func_lengths) / len(func_lengths) if func_lengths else 0.0
    avg_var_len = sum(var_lengths) / len(var_lengths) if var_lengths else 0.0
    avg_file_len = sum(file_name_lengths) / len(file_name_lengths) if file_name_lengths else 0.0
    return (avg_func_len, avg_var_len, avg_file_len)

def aggregate_results(length_dict):
    result = {}
    for quarter in sorted(length_dict.keys()):
        result[quarter] = {}
        for cat in ['cs', 'non_cs']:
            data = length_dict[quarter][cat]
            if not data:
                result[quarter][cat] = {'avg_func_len': 0.0, 'avg_var_len': 0.0, 'avg_file_len': 0.0}
            else:
                func_total = sum((x[0] for x in data))
                var_total = sum((x[1] for x in data))
                file_total = sum((x[2] for x in data))
                count = len(data)
                result[quarter][cat] = {'avg_func_len': round(func_total / count, 4), 'avg_var_len': round(var_total / count, 4), 'avg_file_len': round(file_total / count, 4)}
    return result
base_dir = f'LLM_code/arxiv_dataset'
output_dir = 'LLM_code/arxiv_result/naming_patterns_python'
os.makedirs(output_dir, exist_ok=True)
categories_file = 'LLM_code/code/github_links/python_dataset_links_1.json'
with open(categories_file, 'r', encoding='utf-8') as f:
    all_categories = json.load(f)
quarter_repo_category = defaultdict(dict)
for (quarter, items) in all_categories.items():
    for item in items:
        link = item['link']
        category = item['categories']
        repo_name = link.rstrip('/').split('/')[-1]
        quarter_repo_category[quarter][repo_name] = classify_category(category)
quarter_avg_lengths = defaultdict(lambda : defaultdict(list))
for year in range(2020, 2026):
    max_q = 1 if year == 2025 else 4
    for q in range(1, max_q + 1):
        quarter = f'{year}Q{q}'
        quarter_dir = os.path.join(base_dir, str(year), f'Q{q}')
        if not os.path.isdir(quarter_dir):
            continue
        projects = [d for d in os.listdir(quarter_dir) if os.path.isdir(os.path.join(quarter_dir, d))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_project, os.path.join(quarter_dir, p)): p for p in projects}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f'Scanning {quarter}'):
                proj = futures[future]
                (func_len, var_len, file_len) = future.result()
                category = quarter_repo_category.get(quarter, {}).get(proj)
                if category:
                    quarter_avg_lengths[quarter][category].append((func_len, var_len, file_len))
avg_result = aggregate_results(quarter_avg_lengths)
out_path = os.path.join(output_dir, f'average_lengths_cs_split.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(avg_result, f, ensure_ascii=False, indent=2)