import os
import json
import shutil

year = "2022"
# 定义仓库存储的根目录
repo_base_path = f"github_code/{year}"

# 读取 repo_check_result.json 文件
json_file_path = f"repo_check_{year}.json"

# 确保 JSON 文件存在
if not os.path.exists(json_file_path):
    print(f"Error: JSON file '{json_file_path}' not found.")
    exit(1)

# 读取 JSON 数据
with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 获取需要删除的文件夹列表
extra_folders = data.get("extra_folders", [])

# 删除多余的文件夹
for folder in extra_folders:
    folder_path = os.path.join(repo_base_path, folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)  # 递归删除整个文件夹
            print(f"Deleted folder: {folder}")
        except Exception as e:
            print(f"Failed to delete {folder}: {e}")
    else:
        print(f"Skipping: {folder} (not found)")

print("Cleanup complete.")
