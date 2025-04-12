import os
import requests
import json
from tqdm import tqdm

# === 推荐通过环境变量设置 GitHub Token ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "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")

# 年份和季度（可自由调整）
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]
SEASONS = ["Q1", "Q2", "Q3", "Q4"]

def download_file(url, save_path):
    """下载单个文件"""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download: {url} (Status Code: {response.status_code})")

def get_file_timestamp(repo_owner, repo_name, file_path, branch):
    """获取文件最后更新时间"""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&per_page=1&sha={branch}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200 and response.json():
        return response.json()[0]["commit"]["committer"]["date"]
    return "Unknown"

def get_default_branch(repo_owner, repo_name):
    """获取默认分支（通常是main或master）"""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json().get("default_branch", "main")
    return "main"

def fetch_files_from_github(repo_owner, repo_name, base_dir, language="python", branch=None):
    """根据语言类型爬取对应的文件（Python 或 C/C++）"""
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

    if language == "python":
        exts = [".py"]
        time_info_file = "time_info.txt"
    elif language == "cpp":
        exts = [".c", ".cpp"]
        time_info_file = "time_info_cpp.txt"
    else:
        raise ValueError(f"Unsupported language: {language}")

    tree_items = [
        file for file in response.json().get("tree", [])
        if file["type"] == "blob" and any(file["path"].endswith(ext) for ext in exts)
    ]

    if not tree_items:
        print(f"No {language} files found in {repo_name}.")
        return

    timestamps = []

    for file in tqdm(tree_items, desc=f"Downloading {repo_name} ({language})", leave=False):
        raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
        save_path = os.path.join(save_dir, file["path"])
        download_file(raw_url, save_path)
        last_modified = get_file_timestamp(repo_owner, repo_name, file["path"], branch)
        timestamps.append(f"{file['path']}: {last_modified}")

    with open(os.path.join(save_dir, time_info_file), "w", encoding="utf-8") as f:
        f.write("\n".join(timestamps))

    print(f"{language.capitalize()} timestamps saved in: {os.path.join(save_dir, time_info_file)}")

def process_github_links(links, base_dir, language="python"):
    """处理 GitHub 仓库链接，支持 Python 或 C/C++ 文件抓取"""
    for link in tqdm(links, desc=f"Processing Repositories ({language})"):
        repo_owner, repo_name = link.split("/")[-2], link.split("/")[-1]
        save_dir = os.path.join(base_dir, repo_name)

        if language == "python":
            time_file = "time_info.txt"
        elif language == "cpp":
            time_file = "time_info_cpp.txt"
        else:
            raise ValueError(f"Unsupported language: {language}")

        if os.path.exists(os.path.join(save_dir, time_file)):
            tqdm.write(f"Skipping {repo_name} as {time_file} already exists.")
            continue

        tqdm.write(f"Processing {repo_name} ({language})...")
        fetch_files_from_github(repo_owner, repo_name, base_dir, language=language, branch=None)

# === 主循环 ===

# 处理 Python
# for year in YEARS:
#     for season in SEASONS:
#         BASE_DIR = f"LLM_code/arxiv_dataset/{year}/{season}"
#         json_path_py = f"target/target_{year}{season}.json"

#         if os.path.exists(json_path_py):
#             print(f"\n=== Processing {year} {season} (Python) ===")
#             with open(json_path_py, 'r') as f:
#                 links = json.load(f)
#             process_github_links(links, BASE_DIR, language="python")

# 处理 C/C++
cpp_json_path = "LLM_code/code/github_links/cpp_repos.json"
if os.path.exists(cpp_json_path):
    with open(cpp_json_path, "r") as f:
        cpp_data = json.load(f)

    for quarter, links in cpp_data.items():
        year, season = quarter[:4], quarter[4:]
        BASE_DIR = f"LLM_code/arxiv_dataset/{year}/{season}"
        print(f"\n=== Processing {quarter} (C/C++) ===")
        process_github_links(links, BASE_DIR, language="cpp")
else:
    print("cpp_repos.json not found.")
