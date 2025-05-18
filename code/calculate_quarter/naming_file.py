import os
import re
import json
import warnings
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
warnings.filterwarnings('ignore', category=SyntaxWarning)
naming_patterns = {'single_letter': '^[a-zA-Z]$', 'lowercase': '^[a-z]+$', 'UPPERCASE': '^[A-Z]+$', 'camelCase': '^[a-z]+(?:[A-Z][a-z]*)*$', 'snake_case': '^[a-z]+(?:_[a-z]+)+$', 'PascalCase': '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', 'UPPER_SNAKE_CASE': '^[A-Z]+(?:_[A-Z]+)+$', 'endsWithDigits': '^[A-Za-z_]+[0-9]+$', 'Other': '.*'}

def get_naming_pattern(name):
    """判断命名规则"""
    name = str(name)
    for (pattern, regex) in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return 'Other'

def classify_category(cat):
    """归类为 cs 或 non_cs"""
    if cat.startswith('cs.'):
        return 'cs'
    else:
        return 'non_cs'

def process_project(project_name, quarter_path, quarter_key, quarter_repo_category):
    """处理单个项目，统计文件名命名规则"""
    project_path = os.path.join(quarter_path, project_name)
    project_category = quarter_repo_category.get(quarter_key, {}).get(project_name)
    if project_category is None:
        return None
    local_filename_counts = defaultdict(int)
    for (root, _, files) in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                filename_without_ext = os.path.splitext(file)[0]
                pattern = get_naming_pattern(filename_without_ext)
                local_filename_counts[project_category, pattern] += 1
    return local_filename_counts
base_dir = 'LLM_code/arxiv_dataset'
output_dir = 'LLM_code/arxiv_result/naming_patterns_python'
categories_file = 'LLM_code/code/github_links/python_dataset_links_1.json'
os.makedirs(output_dir, exist_ok=True)
with open(categories_file, 'r', encoding='utf-8') as f:
    all_categories = json.load(f)
quarter_repo_category = defaultdict(dict)
for (quarter, items) in all_categories.items():
    for item in items:
        link = item['link']
        categories = item['categories']
        repo_name = link.rstrip('/').split('/')[-1]
        quarter_repo_category[quarter][repo_name] = classify_category(categories)
quarter_filename_counts = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
for year in range(2020, 2026):
    max_quarter = 1 if year == 2025 else 4
    for q in range(1, max_quarter + 1):
        quarter_name = f'Q{q}'
        year_str = str(year)
        quarter_key = f'{year_str}Q{q}'
        quarter_path = os.path.join(base_dir, year_str, quarter_name)
        if not os.path.isdir(quarter_path):
            continue
        project_list = [d for d in os.listdir(quarter_path) if os.path.isdir(os.path.join(quarter_path, d))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for project_name in project_list:
                futures.append(executor.submit(process_project, project_name, quarter_path, quarter_key, quarter_repo_category))
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f'Scanning {quarter_key}'):
                result = future.result()
                if result is None:
                    continue
                for ((project_category, pattern), count) in result.items():
                    quarter_filename_counts[quarter_key][project_category][pattern] += count
all_categories_list = ['cs', 'non_cs']

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
final_filename_output = compute_ratios(quarter_filename_counts)
filename_output_path = os.path.join(output_dir, 'naming_patterns_filename.json')
with open(filename_output_path, 'w', encoding='utf-8') as f:
    json.dump(final_filename_output, f, ensure_ascii=False, indent=2)