import os
import requests
import json

# GitHub Token（建议用环境变量）
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN",
                         "your_token")

# 所有代码存放的主目录
BASE_DIR = "github_code"


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
        return last_commit["commit"]["committer"]["date"]  # 返回 ISO 时间字符串
    return "Unknown"


def get_default_branch(repo_owner, repo_name):
    """获取仓库的默认分支（master 或 main）"""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json().get("default_branch", "main")  # 默认返回 'main'，如果没有字段
    return "main"


def fetch_python_files_from_github(repo_owner, repo_name, branch=None):
    """使用 GitHub API 获取仓库中的 .py 文件并下载，同时获取更新时间"""
    save_dir = os.path.join(BASE_DIR, repo_name)  # 每个仓库存放到 github_code/repo_name
    os.makedirs(save_dir, exist_ok=True)

    # 如果没有提供分支，则获取默认分支
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
    timestamps = []  # 存储每个文件的更新时间

    for file in data.get("tree", []):
        if file["type"] == "blob" and file["path"].endswith(".py"):
            raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
            save_path = os.path.join(save_dir, file["path"])

            # 下载文件
            download_file(raw_url, save_path)

            # 获取文件更新时间
            last_modified = get_file_timestamp(repo_owner, repo_name, file["path"], branch)
            timestamps.append(f"{file['path']}: {last_modified}")

    # 把所有的时间信息写入文件
    timestamp_file = os.path.join(save_dir, "time_info.txt")
    with open(timestamp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(timestamps))

    print(f"Timestamps saved in: {timestamp_file}")


def process_github_links(json_file):
    """处理 JSON 文件中的 GitHub 链接"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    github_links = data.get("github_links", [])

    for link in github_links:
        # 提取仓库的 owner 和 name
        repo_owner, repo_name = link.split("/")[-2], link.split("/")[-1]
        save_dir = os.path.join(BASE_DIR, repo_name)

        # 如果项目文件夹下已经存在 time_info.txt 文件，跳过
        if os.path.exists(os.path.join(save_dir, "time_info.txt")):
            print(f"Skipping {repo_name} as time_info.txt already exists.")
            continue

        # 否则，获取该仓库的 .py 文件并下载时间戳
        print(f"Processing {repo_name}...")
        fetch_python_files_from_github(repo_owner, repo_name, branch=None)


# 使用示例
process_github_links("link_2020.json")
