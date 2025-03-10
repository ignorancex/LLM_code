import os
import requests

def download_file(url, save_path):
    """下载单个文件"""
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {url}")

def fetch_python_files_from_github(repo_owner, repo_name, branch="master", save_dir="filtered_repo"):
    """使用 GitHub API 获取仓库中的 .py 文件并下载"""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
    headers = {"Accept": "application/vnd.github.v3+json"}

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching repository data: {response.json()}")
        return

    data = response.json()
    for file in data.get("tree", []):
        if file["type"] == "blob" and file["path"].endswith(".py"):
            raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
            save_path = os.path.join(save_dir, file["path"])
            download_file(raw_url, save_path)

# 使用方式：指定 GitHub 仓库的拥有者、仓库名称和分支
fetch_python_files_from_github("mattbierbaum", "arxiv-public-datasets")
