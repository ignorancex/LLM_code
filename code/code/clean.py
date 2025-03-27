import json

# 文件路径
grouped_links_path = 'LLM_code/dataset/github_links/valid_links_by_quarter.json'
still_missing_path = 'still_missing_links.json'
output_cleaned_path = 'LLM_code/dataset/github_links/valid_links_by_quarter.json'

# 加载原始按季度分组的链接
with open(grouped_links_path, 'r', encoding='utf-8') as f:
    grouped_data = json.load(f)

# 加载仍然缺失的链接
with open(still_missing_path, 'r', encoding='utf-8') as f:
    still_missing = json.load(f)

# 构建一个集合，方便快速判断
missing_links_set = set(item['github_link'] for item in still_missing)

# 构建新数据，删除缺失链接
cleaned_grouped_data = {}

for quarter, links in grouped_data.items():
    # 过滤掉 missing 中的链接
    filtered_links = [link for link in links if link not in missing_links_set]
    if filtered_links:
        cleaned_grouped_data[quarter] = filtered_links  # 只保留非空季度

# 保存为新的 JSON 文件
with open(output_cleaned_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned_grouped_data, f, indent=4)

print(f"✅ 清理完成，保存为 {output_cleaned_path}，共 {len(cleaned_grouped_data)} 个季度")
