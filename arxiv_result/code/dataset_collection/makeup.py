import json
import os
import shutil
from urllib.parse import urlparse
missing_links_path = 'missing_links.json'
source_base_dir = 'LLM_code/dataset/github_code'
target_base_dir = 'LLM_code/arxiv_dataset'
output_still_missing = 'still_missing_links.json'
possible_years = [str(y) for y in range(2020, 2026)]
with open(missing_links_path, 'r', encoding='utf-8') as f:
    missing_links = json.load(f)
still_missing = []
for item in missing_links:
    quarter = item['quarter']
    link = item['github_link']
    repo = urlparse(link).path.strip('/').split('/')[-1]
    found = False
    for year in possible_years:
        search_path = os.path.join(source_base_dir, year, repo)
        if os.path.isdir(search_path):
            target_dir = os.path.join(target_base_dir, quarter[:4], quarter[4:])
            os.makedirs(target_dir, exist_ok=True)
            dest_path = os.path.join(target_dir, repo)
            try:
                shutil.move(search_path, dest_path)
                found = True
                break
            except Exception as e:
                break
    if not found:
        still_missing.append(item)
with open(output_still_missing, 'w', encoding='utf-8') as f:
    json.dump(still_missing, f, indent=4)