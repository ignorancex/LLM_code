import json
import re
from tqdm import tqdm

# 读取 JSON 文件
with open("/home/cdp/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/222/arxiv-metadata-oai-snapshot.json", "r", encoding="utf-8") as f:
    json_str = f.read()

# 解析 JSON 对象
json_objects = [json.loads(obj) for obj in json_str.strip().split("\n")]

# 正则表达式匹配 GitHub 链接
# github_pattern = re.compile(r"https://github\.com/[^\s,]+")

github_pattern = re.compile(r"[\(\[\{]?(https://github\.com/[^\s,)\]}\.]+(?:\.[^\s,)\]}\.]+)*)[\)\]}\.]?")

filtered_data = []
total_count = len(json_objects)  # 总数据量

# 使用 tqdm 显示进度
for obj in tqdm(json_objects, desc="Processing records"):
    github_links = set()  # 使用集合去重
    
    # 检查 comments
    if "comments" in obj and isinstance(obj["comments"], str):
        github_links.update(github_pattern.findall(obj["comments"]))
    
    # 检查 abstracts
    if "abstract" in obj and isinstance(obj["abstract"], str):
        github_links.update(github_pattern.findall(obj["abstract"]))

    # 如果找到 GitHub 链接，则存储该项
    if github_links:
        filtered_data.append({
            "id": obj.get("id"),
            "title": obj.get("title"),
            "comments": obj.get("comments"),
            "categories": obj.get("categories"),
            "abstract": obj.get("abstract"),
            "update_date": obj.get("update_date"),
            "github_links": list(github_links)  # 转换为列表
        })

# 保存到新的 JSON 文件
with open("LLM_Code/github_links.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

# 输出统计信息
filtered_count = len(filtered_data)
print(f"Total records: {total_count}")
print(f"Records containing GitHub links: {filtered_count}")
print(f"Retention rate: {filtered_count / total_count * 100:.2f}%")
