import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import time
import os
import concurrent.futures
only_links_path = 'LLM_code/code/github_links/links_with_categories.json'
dataset_links_path = 'LLM_code/code/github_links/python_dataset_links.json'
simulation_links_path = 'LLM_code/code/github_links/python_simulation_links.json'
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')
HEADERS = {'Accept': 'application/vnd.github.v3+json', 'Authorization': f'token {GITHUB_TOKEN}'}
quarters = [f'{year}Q{q}' for year in range(2020, 2026) for q in range(1, 5)]
with open(only_links_path, 'r', encoding='utf-8') as f:
    all_only_links = json.load(f)
with open(dataset_links_path, 'r', encoding='utf-8') as f:
    dataset_links = json.load(f)
with open(simulation_links_path, 'r', encoding='utf-8') as f:
    simulation_links = json.load(f)

def get_default_branch(user, repo):
    url = f'https://api.github.com/repos/{user}/{repo}'
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get('default_branch', 'main')
        else:
            return 'main'
    except Exception as e:
        return 'main'

def has_python_file_fast(user, repo):
    """一次性获取repo的所有文件列表，检查是否包含.py文件"""
    branch = get_default_branch(user, repo)
    url = f'https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1'
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            return False
        files = response.json().get('tree', [])
        for file in files:
            if file['type'] == 'blob' and file['path'].endswith('.py'):
                return True
        return False
    except Exception as e:
        return False
for quarter in quarters:
    only_items = all_only_links.get(quarter, [])
    dataset = dataset_links.get(quarter, [])
    simulation = simulation_links.get(quarter, [])
    existing_links = set(dataset + simulation)
    filtered_links = []
    for item in only_items:
        link = item.get('link', '').strip()
        category = item.get('categories', '').strip()
        if not link:
            continue
        if category.startswith('cs.'):
            continue
        if link not in existing_links:
            filtered_links.append(link)
    if not filtered_links:
        continue
    results = []
    with tqdm(total=len(filtered_links), desc=f'{quarter} Progress', unit='link') as progress_bar:
        for link in tqdm(filtered_links, desc=f'Checking {quarter}', leave=False):
            parts = urlparse(link).path.strip('/').split('/')
            if len(parts) != 2:
                continue
            (user, repo) = parts
            if has_python_file_fast(user, repo):
                results.append(link)
                progress_bar.update(1)
            time.sleep(0.2)
    output_target_path = f'target/target_python_{quarter}.json'
    os.makedirs('target', exist_ok=True)
    with open(output_target_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)