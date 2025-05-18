import json
from tqdm import tqdm
input_path = 'LLM_code/codeforces/cf_code_plain.json'
output_path = 'LLM_code/codeforces/unique_by_problem.json'
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
seen = set()
unique_items = []
for item in tqdm(data, desc='Selecting unique problems'):
    pid = item.get('problem_id')
    if pid not in seen:
        seen.add(pid)
        unique_items.append(item)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(unique_items, f, ensure_ascii=False, indent=2)