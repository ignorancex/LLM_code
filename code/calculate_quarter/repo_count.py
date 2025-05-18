import os
import json
import csv
from collections import defaultdict
json_path = 'LLM_code/code/github_links/cpp_dataset_links_1.json'
root_dir = 'LLM_code/arxiv_dataset_cpp'
output_csv = 'cpp_repo_file_counts.csv'

def classify_category(cat):
    return 'cs' if cat.startswith('cs.') else 'non_cs'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
quarter_repo_category = defaultdict(lambda : defaultdict(list))
for (quarter, repos) in data.items():
    year = quarter[:4]
    season = quarter[4:]
    for entry in repos:
        link = entry['link']
        category = entry['categories'].split()[0]
        category_type = classify_category(category)
        repo_name = link.strip('/').split('/')[-1]
        quarter_repo_category[f'{year}{season}'][category_type].append(repo_name)
results = []
for (quarter, cat_repos) in quarter_repo_category.items():
    year = quarter[:4]
    season = quarter[4:]
    base_path = os.path.join(root_dir, year, season)
    for cat in ['cs', 'non_cs']:
        repo_list = cat_repos.get(cat, [])
        num_files = 0
        for repo in repo_list:
            repo_path = os.path.join(base_path, repo)
            if not os.path.exists(repo_path):
                continue
            for (root, _, files) in os.walk(repo_path):
                num_files += sum((1 for file in files if file.endswith('.c') or file.endswith('.cpp')))
        results.append({'quarter': quarter, 'category': cat, 'num_repos': len(repo_list), 'num_files': num_files})
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['quarter', 'category', 'num_repos', 'num_files'])
    writer.writeheader()
    writer.writerows(results)