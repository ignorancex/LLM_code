import json
from tqdm import tqdm

# 文件路径
input_path = 'LLM_code/codeforces/cf_code_plain.json'         # 原始文件名，包含重复的 problem_id
output_path = 'LLM_code/codeforces/unique_by_problem.json'  # 输出文件名

# 读取 JSON
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 用于去重
seen = set()
unique_items = []

# 按顺序保留每个 problem_id 的第一项
for item in tqdm(data, desc="Selecting unique problems"):
    pid = item.get("problem_id")
    if pid not in seen:
        seen.add(pid)
        unique_items.append(item)

# 写入结果
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(unique_items, f, ensure_ascii=False, indent=2)

print(f"✅ 每个 problem_id 保留一项，已写入 {output_path}")
