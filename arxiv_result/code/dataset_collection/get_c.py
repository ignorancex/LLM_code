import json
import requests
from urllib.parse import urljoin
from tqdm import tqdm
import time
import os
GITHUB_TOKEN = ''
HEADERS = {'Accept': 'application/vnd.github.v3+json', 'Authorization': f'token {GITHUB_TOKEN}'}

def has_cpp_files(repo_url):
    try:
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        user_repo = '/'.join(repo_url.split('/')[-2:])
        api_url = f'https://api.github.com/repos/{user_repo}/git/trees/HEAD?recursive=1'
        response = requests.get(api_url, headers=HEADERS)
        if response.status_code != 200:
            return False
        tree = response.json().get('tree', [])
        for item in tree:
            if item['type'] == 'blob' and (item['path'].endswith('.c') or item['path'].endswith('.cpp')):
                return True
        return False
    except Exception as e:
        return False

def filter_cpp_repos_incremental(input_path, output_path):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            result_data = json.load(f)
    else:
        result_data = {}
    for (quarter, urls) in input_data.items():
        done_urls = set(result_data.get(quarter, []))
        cpp_repos = result_data.get(quarter, [])
        for url in tqdm(urls, desc=f'{quarter}', unit='repo'):
            if url in done_urls:
                continue
            if has_cpp_files(url):
                cpp_repos.append(url)
            result_data[quarter] = cpp_repos
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            time.sleep(0.5)
filter_cpp_repos_incremental('LLM_code/code/github_links/valid_links_by_quarter_new.json', 'LLM_code/code/github_links/cpp_repos.json')