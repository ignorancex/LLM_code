import json
import requests
from datetime import datetime
from tqdm import tqdm

def is_link_accessible(url):
    """检查链接是否可访问"""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# 读取 JSON 文件
with open('LLM_code/dataset/github_links/Only_links.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 设置时间段
start_date = datetime(2020, 1, 1)
end_date = datetime(2020, 3, 31)

# 筛选出在时间范围内的项目
filtered_links = [
    item for item in data
    if start_date <= datetime.strptime(item['update_date'], "%Y-%m-%d") <= end_date
]

# 检查哪些链接是可访问的（添加 tqdm 进度条）
accessible_items = []
for item in tqdm(filtered_links, desc="Checking GitHub links"):
    url = item.get('github_links')
    if url and is_link_accessible(url):
        accessible_items.append(item)

# 保存可访问链接到新的 JSON 文件
with open('LLM_code/dataset/github_links/accessible_links_2020Q1.json', 'w', encoding='utf-8') as f:
    json.dump(accessible_items, f, indent=2, ensure_ascii=False)

# 输出统计信息
print(f"\n✅ 可访问的链接数量: {len(accessible_items)}")
