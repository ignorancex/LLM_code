import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import time

# === 配置路径 ===
only_links_path = 'LLM_code/code/github_links/Only_links_by_quarter.json'
valid_links_path = 'LLM_code/code/github_links/valid_links_by_quarter.json'

# === GitHub Token ===
GITHUB_TOKEN = ''  # ✅ 实际使用中建议放到环境变量中

HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'token {GITHUB_TOKEN}'
}

# === 要处理的季度列表 ===
quarters = [
    f"{year}Q{q}" for year in range(2020, 2026)
    for q in range(1, 5) if not (year == 2020 and q == 1) and not (year == 2025 and q > 1)
]

# === 加载链接数据 ===
with open(only_links_path, 'r', encoding='utf-8') as f:
    all_only_links = json.load(f)

with open(valid_links_path, 'r', encoding='utf-8') as f:
    all_valid_links = json.load(f)

# === 递归检查函数 ===
def has_python_file_recursive(user, repo, path=""):
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            return False

        items = response.json()
        for item in items:
            if item["type"] == "file" and item["name"].endswith(".py"):
                return True
            elif item["type"] == "dir":
                if has_python_file_recursive(user, repo, item["path"]):
                    return True
        return False
    except Exception as e:
        print(f"⚠️ Error accessing {url}: {e}")
        return False

# === 主逻辑 ===
for quarter in quarters:
    print(f"\n📦 处理季度：{quarter}")

    only_links = all_only_links.get(quarter, [])
    existing_links = set(all_valid_links.get(quarter, []))
    to_check_links = [link for link in only_links if link not in existing_links]

    needed_count = 500 - len(existing_links)
    if needed_count <= 0:
        print(f"✅ {quarter} 已有 {len(existing_links)} 个有效链接，无需补充。")
        continue

    print(f"🔍 {quarter} 需要补充 {needed_count} 个链接，共可选 {len(to_check_links)} 个")

    results = []
    for link in tqdm(to_check_links, desc=f"Checking {quarter}"):
        parts = urlparse(link).path.strip("/").split("/")
        if len(parts) != 2:
            continue
        user, repo = parts
        if has_python_file_recursive(user, repo):
            results.append(link)
        time.sleep(0.2)
        if len(results) >= needed_count:
            break

    # 保存本季度临时结果
    output_target_path = f'target_{quarter}.json'
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
