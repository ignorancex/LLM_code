import json

# 你自己的5个JSON文件路径
json_files = [
    'LLM_code/dataset/github_links/links_non_empty/link_2020_filtered.json',
    'LLM_code/dataset/github_links/links_non_empty/link_2021_filtered.json',
    'LLM_code/dataset/github_links/links_non_empty/link_2022_filtered.json',
    'LLM_code/dataset/github_links/links_non_empty/link_2023_filtered.json',
    'LLM_code/dataset/github_links/links_non_empty/link_2024_filtered.json',
]

# 用于保存合并后JSON文件的路径
output_path = 'LLM_code/dataset/github_links/links_non_empty.json'

all_links = []

for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            links = data.get('github_links', [])
            all_links.extend(links)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# 可选：去重
all_links = list(set(all_links))

# 写入新的JSON文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({'github_links': all_links}, f, indent=4)

print(f"合并完成，共包含 {len(all_links)} 条链接，保存至 {output_path}")
