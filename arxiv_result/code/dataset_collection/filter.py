import json
import re
from tqdm import tqdm
input_file = 'Code/github_links.json'
output_file_filtered = 'Code/filtered_github_links.json'
output_file_details = 'Code/Only_links.json'
github_pattern = re.compile('^https://github\\.com/([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)/([\\w.-]+)(?:\\.git)?$')
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_data = []
details_data = []
for obj in tqdm(data, desc='Processing records', ncols=80):
    github_link = obj.get('github_links', '').strip().rstrip('/;')
    match = github_pattern.match(github_link)
    if match:
        (user, repo) = match.groups()
        if github_link.endswith('.git'):
            github_link = github_link[:-4]
        obj['github_links'] = github_link
        filtered_data.append(obj)
        details_data.append({'github_links': github_link, 'user': user, 'repo': repo, 'update_date': obj.get('update_date', 'N/A')})
with open(output_file_filtered, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
with open(output_file_details, 'w', encoding='utf-8') as f:
    json.dump(details_data, f, ensure_ascii=False, indent=4)
total_details = len(details_data)