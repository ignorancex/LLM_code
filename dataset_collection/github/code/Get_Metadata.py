import json
import re
from tqdm import tqdm
input_file = '/home/cdp/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/222/arxiv-metadata-oai-snapshot.json'
output_file = 'Code/github_links.json'
with open(input_file, 'r', encoding='utf-8') as f:
    json_str = f.read()
json_objects = [json.loads(obj) for obj in json_str.strip().split('\n')]
github_pattern = re.compile('[\\(\\[\\{]?(https://github\\.com/[^\\s,)\\]}\\.]+(?:\\.[^\\s,)\\]}\\.]+)*)[\\)\\]}\\.]?')
filtered_data = []
total_count = len(json_objects)
for obj in tqdm(json_objects, desc='Processing records', ncols=80):
    github_links = set()
    if 'comments' in obj and isinstance(obj['comments'], str):
        github_links.update(github_pattern.findall(obj['comments']))
    if 'abstract' in obj and isinstance(obj['abstract'], str):
        github_links.update(github_pattern.findall(obj['abstract']))
    unique_links = list(github_links)
    if len(unique_links) == 1:
        filtered_data.append({'id': obj.get('id'), 'title': obj.get('title'), 'comments': obj.get('comments'), 'categories': obj.get('categories'), 'abstract': obj.get('abstract'), 'update_date': obj.get('update_date'), 'github_links': unique_links[0]})
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)
filtered_count = len(filtered_data)