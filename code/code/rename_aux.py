import os
import json

# 目录路径
year = "2024"
repo_base_path = f"LLM_code/code_analyze/dataset/github_code/{year}"
output_json_path = f"LLM_code/code_analyze/dataset/rename/renamed_{year}.json"

# 存储重命名的文件
renamed_files = []

# 遍历所有子目录和文件
for root, _, files in os.walk(repo_base_path):
    for file in files:
        if file.lower() == "aux.py":
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, "aux_file.py")
            os.rename(old_path, new_path)
            renamed_files.append({"old_path": old_path, "new_path": new_path})

# 保存重命名信息到 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(renamed_files, json_file, indent=4, ensure_ascii=False)

print(f"Renaming complete. {len(renamed_files)} files renamed. Details saved in '{output_json_path}'.")
