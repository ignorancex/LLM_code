import os
import ast
import re
import json
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

# 不启用分类
classify_enabled = False

naming_patterns = {
    'single_letter':       '^[a-zA-Z]$',
    'lowercase':           '^[a-z]+$',
    'UPPERCASE':           '^[A-Z]+$',
    'camelCase':           '^[a-z]+(?:[A-Z][a-z]*)*$',
    'snake_case':          '^[a-z]+(?:_[a-z]+)+$',
    'PascalCase':          '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$',
    'endsWithDigits':      '^[A-Za-z_]+[0-9]+$',
    'Other':               '.*'
}

def get_naming_pattern(name):
    for pattern, regex in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return 'Other'

def extract_code_info(file_path, skipped_files_log):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        if '\x00' in code:
            raise ValueError('Contains null bytes')
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        with open(skipped_files_log, 'a', encoding='utf-8') as log:
            log.write(f'Skipped {file_path}: {str(e)}\n')
        return set(), set()
    func_names = set()
    var_names  = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            func_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    var_names.add(t.id)
    return func_names, var_names

def process_project(project_name, quarter_path, skipped_files_log):
    project_path = os.path.join(quarter_path, project_name)
    # 1. 收集所有 .py 文件，并统计每个文件的行数
    py_files = []
    for root, _, files in os.walk(project_path):
        for fn in files:
            if fn.endswith('.py'):
                path = os.path.join(root, fn)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    py_files.append((path, line_count))
                except Exception:
                    # 计入跳过日志
                    with open(skipped_files_log, 'a', encoding='utf-8') as log:
                        log.write(f'Skipped {path}: could not count lines\n')

    if not py_files:
        return None

    # 2. 按行数排序并对半分
    py_files.sort(key=lambda x: x[1])
    mid = len(py_files) // 2
    fewer_files = [p for p, _ in py_files[:mid]]
    more_files  = [p for p, _ in py_files[mid:]]

    # 3. 定义一个内部统计函数
    def count_patterns(file_list):
        func_counts = defaultdict(int)
        var_counts  = defaultdict(int)
        for fp in file_list:
            funcs, vars_ = extract_code_info(fp, skipped_files_log)
            for name in funcs:
                func_counts[get_naming_pattern(name)] += 1
            for name in vars_:
                var_counts[get_naming_pattern(name)] += 1
        # 归一化为比例
        ftot = sum(func_counts.values()) or 1
        vtot = sum(var_counts.values())  or 1
        func_ratios = {pat: func_counts.get(pat,0)/ftot for pat in naming_patterns}
        var_ratios  = {pat: var_counts.get(pat,0)/vtot for pat in naming_patterns}
        return func_ratios, var_ratios

    fewer_func_r, fewer_var_r = count_patterns(fewer_files)
    more_func_r,  more_var_r  = count_patterns(more_files)

    return {
        'project': project_name,
        'fewer':   { 'func': fewer_func_r, 'var': fewer_var_r },
        'more':    { 'func': more_func_r,  'var': more_var_r  }
    }

# 主流程
base_dir      = 'LLM_code/arxiv_dataset'
output_dir    = 'LLM_code/arxiv_result/naming_patterns_split'
os.makedirs(output_dir, exist_ok=True)
skipped_files = os.path.join(output_dir, 'skipped_files.txt')
if os.path.exists(skipped_files):
    os.remove(skipped_files)

results = []

for year in range(2020, 2026):
    max_q = 1 if year == 2025 else 4
    for q in range(1, max_q+1):
        quarter_key  = f'{year}Q{q}'
        quarter_path = os.path.join(base_dir, str(year), f'Q{q}')
        if not os.path.isdir(quarter_path):
            continue
        projects = [d for d in os.listdir(quarter_path)
                    if os.path.isdir(os.path.join(quarter_path, d))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = { executor.submit(process_project, p, quarter_path, skipped_files): p
                        for p in projects }
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), desc=f'Scanning {quarter_key}'):
                res = fut.result()
                if res:
                    res['quarter'] = quarter_key
                    results.append(res)

# 将结果写入 JSON
out_path = os.path.join(output_dir, 'naming_patterns_split.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done. 统计结果保存在", out_path)
