import json
from collections import defaultdict

# 读取数据
with open("LLM_code/code/github_links/categories.json", "r") as f:
    data = json.load(f)

# 初始化结果字典
summary = defaultdict(lambda: {"cs.CV": 0, "cs.CL": 0, "cs.*": 0, "non-cs": 0})

# 遍历数据并统计
for quarter, entries in data.items():
    for entry in entries:
        cat = entry["categories"]
        if cat == "cs.CV":
            summary[quarter]["cs.CV"] += 1
        if cat == "cs.CL":
            summary[quarter]["cs.CL"] += 1
        if cat.startswith("cs."):
            summary[quarter]["cs.*"] += 1
        else:
            summary[quarter]["non-cs"] += 1

# 输出结果
print(f"{'Quarter':<10} {'cs.CV':>6} {'cs.CL':>6} {'cs.*':>6} {'non-cs':>8}")
print("-" * 40)
for quarter in sorted(summary.keys()):
    counts = summary[quarter]
    print(f"{quarter:<10} {counts['cs.CV']:>6} {counts['cs.CL']:>6} {counts['cs.*']:>6} {counts['non-cs']:>8}")
