import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import time
import concurrent.futures
import os

# === 配置路径 ===
only_links_path = 'LLM_code/code/github_links/Only_links.json'
valid_links_path = 'LLM_code/code/github_links/cpp_repos.json'

# === GitHub Token ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # ✅ 建议用环境变量
HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'token {GITHUB_TOKEN}'
}

# === 要处理的季度列表 ===
quarters = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]

# === 加载链接数据 ===
with open(only_links_path, 'r', encoding='utf-8') as f:
    all_only_links = json.load(f)

with open(valid_links_path, 'r', encoding='utf-8') as f:
    all_valid_links = json.load(f)

# === 获取默认分支 ===
def get_default_branch(user, repo):
    url = f"https://api.github.com/repos/{user}/{repo}"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("default_branch", "main")
        else:
            return "main"
    except Exception as e:
        print(f"⚠️ Error getting default branch for {user}/{repo}: {e}")
        return "main"

# === 新的快速检查函数 ===
def has_c_cpp_file_fast(user, repo):
    """一次性获取repo的所有文件列表，检查是否包含.c或.cpp"""
    branch = get_default_branch(user, repo)
    url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            return False
        files = response.json().get("tree", [])
        for file in files:
            if file["type"] == "blob" and (file["path"].endswith(".c") or file["path"].endswith(".cpp")):
                return True
        return False
    except Exception as e:
        print(f"⚠️ Error accessing {url}: {e}")
        return False

# === 主逻辑 ===
for quarter in quarters:
    print(f"\n📦 处理季度：{quarter}")

    # def center_sorted(lst):
    #     n = len(lst)
    #     mid = (n - 1) / 2
    #     return [x for _, x in sorted(enumerate(lst), key=lambda t: abs(t[0] - mid))]

    only_links = all_only_links.get(quarter, [])
    existing_links = set(all_valid_links.get(quarter, []))
    to_check_links = [link for link in reversed(only_links) if link not in existing_links]

    needed_count = 103 - len(existing_links)
    if quarter in ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]:
        needed_count = 128 - len(existing_links)

    if needed_count <= 0:
        print(f"✅ {quarter} 已有 {len(existing_links)} 个有效链接，无需补充。")
        continue

    print(f"🔍 {quarter} 需要补充 {needed_count} 个链接，共可选 {len(to_check_links)} 个")

    results = []

    with tqdm(total=needed_count, desc=f"{quarter} Progress", unit="link") as progress_bar:
        for link in tqdm(to_check_links, desc=f"Checking {quarter}", leave=False):
            parts = urlparse(link).path.strip("/").split("/")
            if len(parts) != 2:
                continue
            user, repo = parts
            if has_c_cpp_file_fast(user, repo):
                results.append(link)
                progress_bar.update(1)
            time.sleep(0.2)
            if len(results) >= needed_count:
                break

    # 保存本季度临时结果
    output_target_path = f'target/target_{quarter}.json'
    os.makedirs('target', exist_ok=True)
    with open(output_target_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # === 更新 valid_links 数据并保存 ===
    all_valid_links.setdefault(quarter, [])
    all_valid_links[quarter].extend(results)

    print(f"✅ {quarter} 筛选完成，添加 {len(results)} 个链接，保存至 {output_target_path}")

# === 保存更新后的 valid_links_by_quarter.json ===
with open(valid_links_path, 'w', encoding='utf-8') as f:
    json.dump(all_valid_links, f, indent=4)

print("\n📝 所有季度处理完成，valid_links_by_quarter.json 已更新。")
