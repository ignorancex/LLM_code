import json
from pathlib import Path
from collections import defaultdict

# 设置路径
model_dir = Path("LLM_code/codeforces/simulation/output")
dataset_dir = Path("dataset")

# 收集语言对应的模型文件
lang_files = defaultdict(list)
for file in model_dir.glob("*.json"):
    if "_python.json" in file.name:
        lang_files["python"].append(file)
    elif "_cpp.json" in file.name:
        lang_files["cpp"].append(file)

# 处理每种语言
for lang, files in lang_files.items():
    # === 1. 获取每个文件中的 submission_id 集合 ===
    file_to_ids = {}
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            items = json.load(f)
        sid_set = {item["submission_id"] for item in items if "submission_id" in item}
        file_to_ids[file] = sid_set

    # === 2. 求交集 ===
    common_ids = set.intersection(*file_to_ids.values())
    print(f"[{lang.upper()}] 保留的共同 submission_id 数量: {len(common_ids)}")

    # === 3. 过滤每个模型文件，仅保留公共 submission_id 的条目 ===
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            items = json.load(f)
        filtered_items = [item for item in items if item.get("submission_id") in common_ids]
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(filtered_items, f, indent=2, ensure_ascii=False)

    # === 4. 处理 unique_problem_{lang}.json 文件 ===
    unique_path = dataset_dir / f"unique_problem_{lang}.json"
    if unique_path.exists():
        with open(unique_path, 'r', encoding='utf-8') as f:
            unique_items = json.load(f)
        filtered_unique = [item for item in unique_items if item.get("submission_id") in common_ids]
        with open(unique_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_unique, f, indent=2, ensure_ascii=False)
        print(f"[{lang.upper()}] 对应 unique_problem 文件也保留了 {len(filtered_unique)} 条目。")
    else:
        print(f"[{lang.upper()}] 未找到对应的 unique_problem_{lang}.json 文件，跳过。")
