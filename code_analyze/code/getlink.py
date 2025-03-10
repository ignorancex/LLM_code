import json


# 读取 JSON 文件
def extract_github_links(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 过滤 id 以 '20' 开头的条目，并提取 github_links
    filtered_links = []
    for item in data:
        if item.get("id", "").startswith("20") and "github_links" in item:
            filtered_links.extend(item["github_links"])

    # 生成新结构
    output_data = {"github_links": filtered_links}

    # 写入新的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"提取完成，共 {len(filtered_links)} 个 GitHub 链接，已保存至 {output_file}")


# 示例使用
extract_github_links("github_links.json", "link_2020.json")
