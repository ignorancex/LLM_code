import os
import requests
import json
from tqdm import tqdm
import time
import concurrent.futures
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')
YEARS = ['2025']
SEASONS = ['Q1']
LANGUAGE = 'cpp'
USE_TARGET_MODE = True
HEADERS = {'Accept': 'application/vnd.github.v3+json', 'Authorization': f'token {GITHUB_TOKEN}', 'User-Agent': 'Mozilla/5.0'}

def safe_request(url, headers, max_retries=3, timeout=10):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            return response
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise e

def download_file(url, save_path):
    response = safe_request(url, headers=HEADERS)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)

def get_file_timestamp(repo_owner, repo_name, file_path, branch):
    api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&per_page=1&sha={branch}'
    response = safe_request(api_url, headers=HEADERS)
    if response.status_code == 200 and response.json():
        return response.json()[0]['commit']['committer']['date']
    return 'Unknown'

def get_default_branch(repo_owner, repo_name):
    api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}'
    response = safe_request(api_url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get('default_branch', 'main')
    return 'main'

def download_and_get_timestamp(repo_owner, repo_name, branch, save_dir, file):
    raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
    save_path = os.path.join(save_dir, file['path'])
    try:
        download_file(raw_url, save_path)
        last_modified = get_file_timestamp(repo_owner, repo_name, file['path'], branch)
    except Exception as e:
        last_modified = 'Unknown'
    return f"{file['path']}: {last_modified}"

def fetch_files_from_github(repo_owner, repo_name, base_dir, language='python', branch=None):
    save_dir = os.path.join(base_dir, repo_name)
    os.makedirs(save_dir, exist_ok=True)
    if not branch:
        branch = get_default_branch(repo_owner, repo_name)
    api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1'
    response = safe_request(api_url, headers=HEADERS)
    if response.status_code != 200:
        return
    if language == 'python':
        exts = ['.py']
        time_info_file = 'time_info.txt'
    elif language == 'cpp':
        exts = ['.c', '.cpp']
        time_info_file = 'time_info_cpp.txt'
    else:
        raise ValueError(f'Unsupported language: {language}')
    tree_items = [file for file in response.json().get('tree', []) if file['type'] == 'blob' and any((file['path'].endswith(ext) for ext in exts))]
    MAX_FILE_THRESHOLD = 200
    if len(tree_items) > MAX_FILE_THRESHOLD:
        return
    if not tree_items:
        return
    timestamps = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_and_get_timestamp, repo_owner, repo_name, branch, save_dir, file) for file in tree_items]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f'Downloading {repo_name} ({language})', dynamic_ncols=True, leave=False):
            timestamps.append(f.result())
    with open(os.path.join(save_dir, time_info_file), 'w', encoding='utf-8') as f:
        f.write('\n'.join(timestamps))

def process_github_links(links, base_dir, language='python'):
    for link in tqdm(links, desc=f'Processing Repositories ({language})', dynamic_ncols=True):
        (repo_owner, repo_name) = (link.split('/')[-2], link.split('/')[-1])
        save_dir = os.path.join(base_dir, repo_name)
        if language == 'python':
            time_file = 'time_info.txt'
        elif language == 'cpp':
            time_file = 'time_info_cpp.txt'
        else:
            raise ValueError(f'Unsupported language: {language}')
        if os.path.exists(os.path.join(save_dir, time_file)):
            tqdm.write(f'Skipping {repo_name} as {time_file} already exists.')
            continue
        tqdm.write(f'Processing {repo_name} ({language})...')
        fetch_files_from_github(repo_owner, repo_name, base_dir, language=language, branch=None)
        time.sleep(0.5)

def main():
    for year in YEARS:
        for season in SEASONS:
            BASE_DIR = f'LLM_code/arxiv_dataset_cpp/{year}/{season}'
            if USE_TARGET_MODE:
                json_path = f'LLM_code/code/github_links/target/target_{year}{season}.json'
                if not os.path.exists(json_path):
                    continue
                with open(json_path, 'r') as f:
                    links = json.load(f)
                process_github_links(links, BASE_DIR, language=LANGUAGE)
            else:
                json_path = 'LLM_code/code/github_links/cpp_dataset_links.json'
                if not os.path.exists(json_path):
                    continue
                with open(json_path, 'r') as f:
                    all_links = json.load(f)
                quarter_key = f'{year}Q{season[-1]}'
                links = all_links.get(quarter_key, [])
                if not links:
                    continue
                process_github_links(links, BASE_DIR, language=LANGUAGE)
if __name__ == '__main__':
    while True:
        try:
            main()
            break
        except Exception as e:
            time.sleep(5)
            continue