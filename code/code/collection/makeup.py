import json
import os
import shutil
from urllib.parse import urlparse

# 输入文件路径
missing_links_path = 'missing_links.json'
source_base_dir = 'LLM_code/dataset/github_code'
target_base_dir = 'LLM_code/arxiv_dataset'
output_still_missing = 'still_missing_links.json'

# 指定可能的年份范围（根据你数据的实际情况调整）
possible_years = [str(y) for y in range(2020, 2026)]

# 加载 missing_links
with open(missing_links_path, 'r', encoding='utf-8') as f:
    missing_links = json.load(f)

still_missing = []

for item in missing_links:
    quarter = item['quarter']
    link = item['github_link']
    repo = urlparse(link).path.strip('/').split('/')[-1]
    found = False

    for year in possible_years:
        search_path = os.path.join(source_base_dir, year, repo)
        if os.path.isdir(search_path):
            # 找到了，移动到 arxiv_dataset/{quarter}
            target_dir = os.path.join(target_base_dir, quarter[:4], quarter[4:])
            os.makedirs(target_dir, exist_ok=True)
            dest_path = os.path.join(target_dir, repo)

            try:
                shutil.move(search_path, dest_path)
                print(f"✅ 补救移动成功：{repo} → {quarter}")
                found = True
                break
            except Exception as e:
                print(f"⚠️ 移动失败 {repo}: {e}")
                break

    if not found:
        still_missing.append(item)

# 保存最终仍然缺失的
with open(output_still_missing, 'w', encoding='utf-8') as f:
    json.dump(still_missing, f, indent=4)

print(f"\n📝 处理完成。仍找不到的链接共 {len(still_missing)} 条，已写入 {output_still_missing}")
