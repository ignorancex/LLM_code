import json
import re

# 输入和输出文件
input_file = "/media/sata3/siming/LLM_Code/github_links.json"  
output_file = "/media/sata3/siming/LLM_Code/filtered_github_links.json"

# 正则表达式模式
github_pattern = re.compile(
    r'^https://github\.com/'
    r'([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)/'
    r'([\w.-]+)$'
)

# 读取JSON文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 过滤和修正不符合模式的数据
filtered_data = []
for obj in data:
    github_links = obj.get("github_links", [])
    if isinstance(github_links, list):
        valid_links = []
        for link in github_links:
            link = link.strip().rstrip('/；;')
            if github_pattern.match(link):
                valid_links.append(link)
        if valid_links:
            obj["github_links"] = valid_links
            filtered_data.append(obj)

# 将过滤后的数据保存到新文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"Filtered data saved to {output_file}")