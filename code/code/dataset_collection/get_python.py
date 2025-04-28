import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import time
import os
import concurrent.futures

# === 配置路径 ===
only_links_path = 'LLM_code/code/github_links/links_with_categories.json'
dataset_links_path = 'LLM_code/code/github_links/python_dataset_links.json'
simulation_links_path = 'LLM_code/code/github_links/python_simulation_links.json'

# === GitHub Token ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # ✅ 建议用环境变量
HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'token {GITHUB_TOKEN}'
}

# === 要处理的季度列表 ===
quarters = [f"{year}Q{q}" for year in range(2020, 2026) for q in range(1, 5)]

# === 加载链接数据 ===
with open(only_links_path, 'r', encoding='utf-8') as f:
    all_only_links = json.load(f)

with open(dataset_links_path, 'r', encoding='utf-8') as f:
    dataset_links = json.load(f)

with open(simulation_links_path, 'r', encoding='utf-8') as f:
    simulation_links = json.load(f)

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
def has_python_file_fast(user, repo):
    """一次性获取repo的所有文件列表，检查是否包含.py文件"""
    branch = get_default_branch(user, repo)
    url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            return False
        files = response.json().get("tree", [])
        for file in files:
            if file["type"] == "blob" and file["path"].endswith(".py"):
                return True
        return False
    except Exception as e:
        print(f"⚠️ Error accessing {url}: {e}")
        return False

# === 主逻辑 ===
for quarter in quarters:
    print(f"\n📦 处理季度：{quarter}")

    only_items = all_only_links.get(quarter, [])
    
    # ✅ existing_links 合并
    dataset = dataset_links.get(quarter, [])
    simulation = simulation_links.get(quarter, [])
    existing_links = set(dataset + simulation)

    # ✅ 根据类别筛选链接
    filtered_links = []
    for item in only_items:
        link = item.get("link", "").strip()
        category = item.get("categories", "").strip()
        if not link:
            continue
        if category.startswith("cs."):
            continue  # 跳过cs类别
        if link not in existing_links:
            filtered_links.append(link)

    if not filtered_links:
        print(f"✅ {quarter} 无需要处理的链接，跳过。")
        continue

    print(f"🔍 {quarter} 需要检查 {len(filtered_links)} 个链接")

    results = []

    with tqdm(total=len(filtered_links), desc=f"{quarter} Progress", unit="link") as progress_bar:
        for link in tqdm(filtered_links, desc=f"Checking {quarter}", leave=False):
            parts = urlparse(link).path.strip("/").split("/")
            if len(parts) != 2:
                continue
            user, repo = parts
            if has_python_file_fast(user, repo):
                results.append(link)
                progress_bar.update(1)
            time.sleep(0.2)

    # 保存本季度临时结果
    output_target_path = f'target/target_python_{quarter}.json'
    os.makedirs('target', exist_ok=True)
    with open(output_target_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"✅ {quarter} 筛选完成，保存 {len(results)} 个链接到 {output_target_path}")

print("\n📝 所有季度处理完成。")
