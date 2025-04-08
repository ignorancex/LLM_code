import json
import os
import shutil
from urllib.parse import urlparse

# 路径设置
grouped_json_path = 'LLM_code/dataset/github_links/valid_links_by_quarter.json'
source_base_dir = 'LLM_code/dataset/github_code'
target_base_dir = 'LLM_code/arxiv_dataset'
missing_links_output = 'missing_links.json'

# 加载按季度分组的链接数据
with open(grouped_json_path, 'r', encoding='utf-8') as f:
    quarter_links = json.load(f)

missing_links = []

# 遍历每个季度
for quarter, links in quarter_links.items():
    year = quarter[:4]
    quarter_name = quarter[4:]  # "Q1" 这样的字符串
    target_dir = os.path.join(target_base_dir, year, quarter_name)

    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    for link in links:
        try:
            # 从链接中提取 repo 名（https://github.com/user/repo）
            repo = urlparse(link).path.strip('/').split('/')[-1]
            source_path = os.path.join(source_base_dir, year, repo)
            dest_path = os.path.join(target_dir, repo)

            if os.path.isdir(source_path):
                shutil.move(source_path, dest_path)
                print(f"✅ 移动 {repo} → {year}/{quarter_name}")
            else:
                print(f"❌ 找不到文件夹：{source_path}")
                missing_links.append({
                    "quarter": quarter,
                    "github_link": link
                })

        except Exception as e:
            print(f"⚠️ 处理出错：{link}：{e}")
            missing_links.append({
                "quarter": quarter,
                "github_link": link,
                "error": str(e)
            })

# 保存找不到的链接到 JSON
with open(missing_links_output, 'w', encoding='utf-8') as f:
    json.dump(missing_links, f, indent=4)

print(f"\n📁 所有处理完成。共找不到 {len(missing_links)} 个链接，已写入 {missing_links_output}")
