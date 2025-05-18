import os
import json
from collections import defaultdict
input_json_path = 'LLM_code/code/github_links/cpp_dataset_links_new.json'
output_json_path = 'LLM_code/code/github_links/cpp_dataset_links_1.json'
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
new_data = {}
duplicate_repos = defaultdict(list)
for (key, url_dicts) in data.items():
    year = key[:4]
    season = key[4:]
    base_path = os.path.join('LLM_code/arxiv_dataset_cpp', year, season)
    seen = set()
    cleaned_url_dicts = []
    for url_dict in url_dicts:
        link = url_dict.get('link')
        if not link:
            continue
        parts = link.rstrip('/').split('/')
        if len(parts) < 2:
            continue
        repo_name = parts[-1]
        repo_path = os.path.join(base_path, repo_name)
        if os.path.exists(repo_path):
            cleaned_url_dicts.append(url_dict)
            if repo_name in seen:
                duplicate_repos[key].append(repo_name)
            seen.add(repo_name)
    if cleaned_url_dicts:
        new_data[key] = cleaned_url_dicts
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)
if duplicate_repos:
    for (key, names) in duplicate_repos.items():