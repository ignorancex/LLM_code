import json
from collections import defaultdict
from datetime import datetime

# 文件路径
input_file = 'LLM_code/dataset/github_links/Only_links.json'
output_file = 'LLM_code/dataset/github_links/Only_links_set.json'

# 读取 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 分组：将相同 github_links 的项放在一起
link_groups = defaultdict(list)
for item in data:
    link = item.get('github_links')
    if link:
        link_groups[link].append(item)

# 新的去重后结果
deduplicated_data = []

for link, items in link_groups.items():
    if len(items) == 1:
        # 只有一个，不重复，直接保留
        deduplicated_data.append(items[0])
    else:
        # 多个，检查日期差
        try:
            dates = [datetime.strptime(item['update_date'], "%Y-%m-%d") for item in items]
            max_date = max(dates)
            min_date = min(dates)
            delta_days = (max_date - min_date).days

            if delta_days <= 365:
                # 相差不超过一年，保留最新时间的记录
                latest_item = max(items, key=lambda x: x['update_date'])
                deduplicated_data.append(latest_item)
            else:
                # 超过一年 → 全部丢弃
                continue
        except Exception as e:
            print(f"⚠️ 日期解析失败，跳过 {link}：{e}")

# 保存到新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(deduplicated_data, f, indent=4)

print(f"✅ 去重完成：保留 {len(deduplicated_data)} 条记录，已保存到 {output_file}")
