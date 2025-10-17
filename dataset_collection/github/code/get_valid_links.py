import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import time
import os
from datetime import datetime

only_links_path = 'dataset_collection/github/links/new_github_links.json'
dataset_links_path = 'dataset_collection/github/links/cpp_dataset_links.json'

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')

HEADERS = {'Accept': 'application/vnd.github.v3+json'}
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'token {GITHUB_TOKEN}'

quarters = ['2025Q3']

def date_to_quarter(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}Q{q}"
    except Exception:
        return None

def get_default_branch(user, repo):
    url = f'https://api.github.com/repos/{user}/{repo}'
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get('default_branch', 'main')
        else:
            return 'main'
    except Exception:
        return 'main'

def has_target_file(user, repo, extensions):
    branch = get_default_branch(user, repo)
    url = f'https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1'
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            return False
        files = response.json().get('tree', [])
        for file in files:
            if file['type'] == 'blob' and any(file['path'].endswith(ext) for ext in extensions):
                return True
        return False
    except Exception:
        return False



with open(only_links_path, 'r', encoding='utf-8') as f:
    all_only_links = json.load(f)

with open(dataset_links_path, 'r', encoding='utf-8') as f:
    dataset_links = json.load(f)

existing_links = set(dataset_links.get('github_links', []))

tasks = {
    "python": ['.py']
    # 'c':['.c', '.cpp']
}

for quarter in quarters:
    only_links = [
        item['github_links'] for item in all_only_links
        if date_to_quarter(item.get('update_date')) == quarter
    ]
    to_check_links = [link for link in only_links if link not in existing_links]

    for task_name, exts in tasks.items():
        output_dir = 'target'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'target_{quarter}_{task_name}.jsonl')

       
        last_link = None
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    try:
                        data = json.loads(line.strip())
                        last_link = data.get("link", None)
                    except:
                        continue

        start_index = 0
        if last_link and last_link in to_check_links:
            start_index = to_check_links.index(last_link) + 1  

        remaining_links = to_check_links[start_index:]

        with tqdm(total=len(remaining_links), desc=f'{quarter}-{task_name} Progress', unit='link') as progress_bar, \
             open(output_file, 'a', encoding='utf-8') as fout:

            for link in remaining_links:
                parts = urlparse(link).path.strip('/').split('/')
                if len(parts) != 2:
                    progress_bar.update(1)
                    continue
                user, repo = parts
                if has_target_file(user, repo, exts):
                    fout.write(json.dumps({"link": link}, ensure_ascii=False) + "\n")
                    fout.flush()
                progress_bar.update(1)
                time.sleep(0.2)


