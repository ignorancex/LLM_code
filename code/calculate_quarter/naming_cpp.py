"""
统计 C / C++ 仓库中函数名与变量名的命名方式比例
=================================================

依赖:
  pip install tree_sitter tqdm
  # 然后在本目录执行一次:
  python build_lang_so.py
  (见下方 build_lang_so.py 说明)

数据目录结构与原 Python 版本脚本完全一致, 只是结果保存在
LLM_code/arxiv_result/naming_patterns_c_cpp 目录下。
"""
import os
import re
import json
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tree_sitter import Language, Parser
warnings.filterwarnings('ignore', category=SyntaxWarning)
TS_LIB_PATH = 'new/my-languages.so'
if not os.path.exists(TS_LIB_PATH):
    raise RuntimeError(f'找不到 {TS_LIB_PATH}\n请先运行 build_lang_so.py (脚本见本文末尾) 编译 tree-sitter-c & tree-sitter-cpp')
C_LANGUAGE = Language(TS_LIB_PATH, 'c')
CPP_LANGUAGE = Language(TS_LIB_PATH, 'cpp')
PARSER_C = Parser()
PARSER_C.set_language(C_LANGUAGE)
PARSER_CPP = Parser()
PARSER_CPP.set_language(CPP_LANGUAGE)
naming_patterns = {'single_letter': '^[a-zA-Z]$', 'lowercase': '^[a-z]+$', 'UPPERCASE': '^[A-Z]+$', 'camelCase': '^[a-z]+(?:[A-Z][a-z]*)*$', 'snake_case': '^[a-z]+(?:_[a-z]+)+$', 'PascalCase': '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', 'UPPER_SNAKE_CASE': '^[A-Z]+(?:_[A-Z]+)+$', 'endsWithDigits': '^[A-Za-z_]+[0-9]+$', 'Other': '.*'}

def get_naming_pattern(name: str) -> str:
    for (pattern, regex) in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return 'Other'

def extract_code_info(file_path: str, skipped_files_log: str):
    """
    解析 *单个* C / C++ 源文件, 返回:
        (set(function_names), set(variable_names))
    若解析失败, 记录到 skipped_files_log, 并返回空集合
    """
    try:
        with open(file_path, 'rb') as f:
            code_bytes = f.read()
        if b'\x00' in code_bytes:
            raise ValueError('Contains null bytes')
    except Exception as e:
        with open(skipped_files_log, 'a', encoding='utf-8') as log:
            log.write(f'Skipped {file_path}: {e}\n')
        return (set(), set())
    ext = os.path.splitext(file_path)[1].lower()
    parser = PARSER_CPP if ext in {'.cpp', '.c'} else PARSER_C
    try:
        tree = parser.parse(code_bytes)
    except Exception as e:
        with open(skipped_files_log, 'a', encoding='utf-8') as log:
            log.write(f'Skipped {file_path}: parser error {e}\n')
        return (set(), set())
    (func_names, var_names) = (set(), set())
    root_node = tree.root_node

    def traverse(node, parent_type=None, grandparent_type=None):
        cur_type = node.type
        if cur_type == 'identifier':
            name = code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            if parent_type in {'function_declarator'} or grandparent_type in {'function_definition'}:
                func_names.add(name)
            elif parent_type in {'init_declarator', 'field_declarator', 'pointer_declarator', 'array_declarator', 'parameter_declaration', 'reference_declarator'} or grandparent_type in {'declaration'}:
                var_names.add(name)
        for child in node.children:
            traverse(child, cur_type, parent_type)
    try:
        traverse(root_node)
    except RecursionError:
        with open(skipped_files_log, 'a', encoding='utf-8') as log:
            log.write(f'Skipped {file_path}: recursion depth exceeded\n')
        return (set(), set())
    return (func_names, var_names)

def classify_category(cat: str) -> str:
    return 'cs' if cat.startswith('cs.') else 'non_cs'

def process_project(project_name, quarter_path, quarter_key, quarter_repo_category, skipped_files_log):
    project_path = os.path.join(quarter_path, project_name)
    project_category = quarter_repo_category.get(quarter_key, {}).get(project_name)
    if project_category is None:
        return None
    func_counts = defaultdict(int)
    var_counts = defaultdict(int)
    for (root, _, files) in os.walk(project_path):
        for file in files:
            if file.endswith(('.c', '.cpp')):
                fpath = os.path.join(root, file)
                (funcs, vars_) = extract_code_info(fpath, skipped_files_log)
                for name in funcs:
                    func_counts[get_naming_pattern(name)] += 1
                for name in vars_:
                    var_counts[get_naming_pattern(name)] += 1
    func_total = sum(func_counts.values())
    var_total = sum(var_counts.values())
    func_ratios = {p: func_counts[p] / func_total if func_total else 0.0 for p in naming_patterns}
    var_ratios = {p: var_counts[p] / var_total if var_total else 0.0 for p in naming_patterns}
    return (project_category, func_ratios, var_ratios)
BASE_DIR = 'LLM_code/arxiv_dataset_cpp'
OUTPUT_DIR = 'LLM_code/arxiv_result/naming_patterns_c_cpp'
CATEGORIES_JS = 'LLM_code/code/github_links/cpp_dataset_links_new.json'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SKIPPED_LOG = os.path.join(OUTPUT_DIR, 'skipped_files.txt')
if os.path.exists(SKIPPED_LOG):
    os.remove(SKIPPED_LOG)
with open(CATEGORIES_JS, 'r', encoding='utf-8') as f:
    all_categories = json.load(f)
quarter_repo_category = defaultdict(dict)
for (quarter, items) in all_categories.items():
    for item in items:
        repo_name = item['link'].rstrip('/').split('/')[-1]
        quarter_repo_category[quarter][repo_name] = classify_category(item['categories'])
quarter_func_ratios = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
quarter_var_ratios = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
for year in range(2020, 2026):
    max_q = 1 if year == 2025 else 4
    for q in range(1, max_q + 1):
        quarter_key = f'{year}Q{q}'
        quarter_path = os.path.join(BASE_DIR, str(year), f'Q{q}')
        if not os.path.isdir(quarter_path):
            continue
        projects = [d for d in os.listdir(quarter_path) if os.path.isdir(os.path.join(quarter_path, d))]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(process_project, p, quarter_path, quarter_key, quarter_repo_category, SKIPPED_LOG) for p in projects]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f'Scanning {quarter_key}'):
                result = fut.result()
                if result is None:
                    continue
                (cat, f_ratios, v_ratios) = result
                for (pat, r) in f_ratios.items():
                    quarter_func_ratios[quarter_key][cat][pat].append(r)
                for (pat, r) in v_ratios.items():
                    quarter_var_ratios[quarter_key][cat][pat].append(r)

def aggregate(qcv):
    res = {}
    for q in sorted(qcv):
        res[q] = {}
        for cat in ('cs', 'non_cs'):
            res[q][cat] = {}
            for pat in naming_patterns:
                ratios = qcv[q][cat][pat]
                res[q][cat][pat] = round(sum(ratios) / len(ratios), 6) if ratios else 0.0
    return res
final_func = aggregate(quarter_func_ratios)
final_var = aggregate(quarter_var_ratios)
os.makedirs(OUTPUT_DIR, exist_ok=True)
FUNC_JSON = os.path.join(OUTPUT_DIR, 'naming_patterns_function.json')
VAR_JSON = os.path.join(OUTPUT_DIR, 'naming_patterns_variable.json')
with open(FUNC_JSON, 'w', encoding='utf-8') as f:
    json.dump(final_func, f, ensure_ascii=False, indent=2)
with open(VAR_JSON, 'w', encoding='utf-8') as f:
    json.dump(final_var, f, ensure_ascii=False, indent=2)