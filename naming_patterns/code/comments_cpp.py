import os
import re
import json
import csv
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)


def load_repo_field_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    repo_field_map = {}
    for quarter, entries in data.items():
        for item in entries:
            try:
                repo_url = item['link']
                repo_name = repo_url.rstrip('/').split('/')[-1]
                category = item.get('categories', '')
                if category.startswith('cs.'):
                    repo_field_map[repo_name] = 'cs'
                else:
                    repo_field_map[repo_name] = 'non-cs'
            except Exception:
                continue  
    return repo_field_map


repo_field_map = load_repo_field_mapping('dataset_collection/github/links/cpp_dataset_links.json')


def count_comment_and_total_lines_c(file_path):
    comment_lines = set()
    total_lines = 0
    in_multiline_comment = False

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for lineno, line in enumerate(f, 1):
                line_strip = line.strip()
                total_lines += 1

                if in_multiline_comment:
                    comment_lines.add(lineno)
                    if '*/' in line_strip:
                        in_multiline_comment = False
                    continue

                if line_strip.startswith('//') or '//' in line_strip:
                    comment_lines.add(lineno)
                elif '/*' in line_strip:
                    comment_lines.add(lineno)
                    if '*/' not in line_strip:
                        in_multiline_comment = True
    except Exception:
        return 0, 0

    return len(comment_lines), total_lines


def compute_quarter_avg_comment_ratio(quarter_path):
    cs_ratios = []
    noncs_ratios = []

    for repo in os.listdir(quarter_path):
        repo_path = os.path.join(quarter_path, repo)
        if not os.path.isdir(repo_path):
            continue

        repo_comment = 0
        repo_total = 0

        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.c', '.cpp')):
                    file_path = os.path.join(root, file)
                    c_lines, t_lines = count_comment_and_total_lines_c(file_path)
                    repo_comment += c_lines
                    repo_total += t_lines

        if repo_total > 0:
            ratio = repo_comment / repo_total
            field = repo_field_map.get(repo, 'unknown')
            if field == 'cs':
                cs_ratios.append(ratio)
            elif field == 'non-cs':
                noncs_ratios.append(ratio)

    cs_avg = sum(cs_ratios) / len(cs_ratios) if cs_ratios else 0.0
    noncs_avg = sum(noncs_ratios) / len(noncs_ratios) if noncs_ratios else 0.0
    return cs_avg, noncs_avg


base_dir = 'arxiv_dataset_cpp'
results = {}

for year in sorted(os.listdir(base_dir)):
    if not year.isdigit() or not 2020 <= int(year) <= 2025:
        continue

    year_path = os.path.join(base_dir, year)
    if not os.path.isdir(year_path):
        continue

    for quarter in sorted(os.listdir(year_path)):
        quarter_path = os.path.join(year_path, quarter)
        if not os.path.isdir(quarter_path):
            continue

        quarter_key = f'{year}_{quarter}'
        cs_avg, noncs_avg = compute_quarter_avg_comment_ratio(quarter_path)
        results[quarter_key] = {'cs': cs_avg, 'noncs': noncs_avg}

csv_path = 'naming_patterns/github_result/naming_patterns_cpp/plots_cpp_linear_1/comment_ratio_cpp.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Quarter', 'CS_Comment_Ratio', 'NonCS_Comment_Ratio'])
    for q, r in results.items():
        writer.writerow([q, f"{r['cs']:.4f}", f"{r['noncs']:.4f}"])
