import os
import json

# 设置起止年份和季度
start_year = 2020
end_year = 2025
end_quarter = 1  # 包含2025Q1

base_dir = "LLM_code/arxiv_dataset"
output_json_path = "renamed_aux.json"

# 存储所有季度的重命名信息
renamed_files = []

for year in range(start_year, end_year + 1):
    for quarter in range(1, 5):
        # 如果是最后一年，只处理到 end_quarter
        if year == end_year and quarter > end_quarter:
            break

        repo_base_path = os.path.join(base_dir, str(year), f"Q{quarter}")

        if not os.path.exists(repo_base_path):
            print(f"Warning: {repo_base_path} does not exist. Skipping.")
            continue

        for root, _, files in os.walk(repo_base_path):
            for file in files:
                if file.lower() == "aux.py":
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, "aux_file.py")
                    os.rename(old_path, new_path)
                    renamed_files.append({"old_path": old_path, "new_path": new_path})

# 保存重命名信息到 JSON 文件
output_dir = os.path.dirname(output_json_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(renamed_files, json_file, indent=4, ensure_ascii=False)

print(f"Renaming complete. {len(renamed_files)} files renamed. Details saved in '{output_json_path}'.")

