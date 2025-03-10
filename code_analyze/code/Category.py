import json
from collections import Counter

# 输入和输出文件
input_file = "/media/sata3/siming/LLM_Code/github_links.json"  
output_file = "/media/sata3/siming/LLM_Code/category.json"

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 统计类别
category_counter = Counter()

for obj in data:
    categories = obj.get("categories", "").strip()
    if categories:
        first_category = categories.split()[0]  
        category_counter[first_category] += 1

# 将统计结果转换为字典
category_stats = dict(category_counter)

# 保存统计结果到 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(category_stats, f, ensure_ascii=False, indent=4)

print(f"Category statistics saved to {output_file}")
