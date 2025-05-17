import json

# 读取原始 JSON 文件
with open("simulation/unique_gemma27b_cpp.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 用集合存储已出现的 (problem_id, submission_id)
seen = set()
deduplicated_data = []

for item in data:
    key = (item.get("problem_id"), item.get("submission_id"))
    if key not in seen:
        seen.add(key)
        deduplicated_data.append(item)

# 写入去重后的结果
with open("simulation/unique_gemma_27b_cpp.json", "w", encoding="utf-8") as f:
    json.dump(deduplicated_data, f, indent=4, ensure_ascii=False)
