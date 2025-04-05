import os
import requests
import json
from tqdm import tqdm

# GitHub Token（建议用环境变量）
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "YOUR_TOKEN")

SEASONS = ["Q1","Q2", "Q3", "Q4"]
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]


def download_file(url, save_path):
    """下载单个 .py 文件"""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download: {url} (Status Code: {response.status_code})")


def get_file_timestamp(repo_owner, repo_name, file_path, branch):
    """获取 GitHub 仓库中某个文件的最后更新时间"""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&per_page=1&sha={branch}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    response = requests.get(api_url, headers=headers)
    if response.status_code == 200 and response.json():
        last_commit = response.json()[0]
        return last_commit["commit"]["committer"]["date"]
    return "Unknown"


def get_default_branch(repo_owner, repo_name):
    """获取仓库的默认分支（master 或 main）"""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json().get("default_branch", "main")
    return "main"


def fetch_python_files_from_github(repo_owner, repo_name, base_dir, branch=None):
    """使用 GitHub API 获取仓库中的 .py 文件并下载，同时获取更新时间"""
    save_dir = os.path.join(base_dir, repo_name)
    os.makedirs(save_dir, exist_ok=True)

    if not branch:
        branch = get_default_branch(repo_owner, repo_name)

    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching repository data: {response.json()}")
        return

    data = response.json()
    tree_items = [file for file in data.get("tree", []) if file["type"] == "blob" and file["path"].endswith(".py")]

    timestamps = []

    for file in tqdm(tree_items, desc=f"Downloading {repo_name}", leave=False):
        raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
        save_path = os.path.join(save_dir, file["path"])

        download_file(raw_url, save_path)
        last_modified = get_file_timestamp(repo_owner, repo_name, file["path"], branch)
        timestamps.append(f"{file['path']}: {last_modified}")

    timestamp_file = os.path.join(save_dir, "time_info.txt")
    with open(timestamp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(timestamps))

    print(f"Timestamps saved in: {timestamp_file}")


def process_github_links(json_file, base_dir):
    """处理 JSON 文件中的 GitHub 链接"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    for link in tqdm(data, desc=f"Processing Repositories from {json_file}"):
        repo_owner, repo_name = link.split("/")[-2], link.split("/")[-1]
        save_dir = os.path.join(base_dir, repo_name)

        if os.path.exists(os.path.join(save_dir, "time_info.txt")):
            tqdm.write(f"Skipping {repo_name} as time_info.txt already exists.")
            continue

        tqdm.write(f"Processing {repo_name}...")
        fetch_python_files_from_github(repo_owner, repo_name, base_dir, branch=None)


# 主循环处理 2021~2024 年的每个季度
for year in YEARS:
    for season in SEASONS:
        BASE_DIR = f"github_code/{year}/{season}"
        json_path = f"target/target_{year}{season}.json"

        if os.path.exists(json_path):
            print(f"\n=== Processing {year} {season} ===")
            process_github_links(json_path, BASE_DIR)
        else:
            print(f"JSON file not found: {json_path}")
