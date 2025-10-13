import json
import re
from tqdm import tqdm
from datetime import datetime

input_file = '/Users/hsm/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/254/arxiv-metadata-oai-snapshot.json'
output_file = 'dataset_collection/github/links/new_github_links.json'

q2q3_start = datetime(2025, 4, 1)
q2q3_end   = datetime(2025, 9, 30)
before2020 = datetime(2020, 1, 1)

github_pattern = re.compile(r"https://github\.com/[^\s\)\]\}]+")
filtered_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc='Processing records', ncols=80):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        date_str = obj.get('update_date')
        if not date_str:
            continue
        try:
            update_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        if not (q2q3_start <= update_date <= q2q3_end or update_date < before2020):
            continue

        github_links = set()
        if isinstance(obj.get('comments'), str):
            github_links.update(github_pattern.findall(obj['comments']))
        if isinstance(obj.get('abstract'), str):
            github_links.update(github_pattern.findall(obj['abstract']))

        unique_links = list(github_links)
        if len(unique_links) == 1:
            filtered_data.append({
                'id': obj.get('id'),
                'title': obj.get('title'),
                'comments': obj.get('comments'),
                'categories': obj.get('categories'),
                'abstract': obj.get('abstract'),
                'update_date': obj.get('update_date'),
                'github_links': unique_links[0]
            })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print(f"Filtered records: {len(filtered_data)}")
