import json

# 读取 metadata 文件
with open("LLM_code/code/github_links/filtered_github_links.json", "r") as f:
    metadata = json.load(f)

# 创建 GitHub 链接到类别的映射，只取第一个类别
link_to_category = {}
for item in metadata:
    link = item.get("github_links", "").strip()
    if link:
        categories = item.get("categories", "").strip()
        if categories:
            first_category = categories.split()[0]
            link_to_category[link] = first_category

# 读取季度链接文件
with open("LLM_code/code/github_links/dataset_links.json", "r") as f:
    quarter_data = json.load(f)

# 创建输出结构
result = {}
for quarter, links in quarter_data.items():
    result[quarter] = []
    for link in links:
        if link in link_to_category:
            result[quarter].append({
                "link": link,
                "categories": link_to_category[link]
            })

# 保存结果到新文件
with open("quarter_links_with_categories.json", "w") as f:
    json.dump(result, f, indent=4)
