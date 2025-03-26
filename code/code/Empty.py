import os 
import json

# 目录和文件路径
year = "2022"
repo_base_path = f"github_code/{year}"
json_file_path = f"link_{year}_new.json"
output_file_path = f"add_{year}.json"

# 读取 JSON 文件
if not os.path.exists(json_file_path):
    print(f"Error: JSON file '{json_file_path}' not found.")
    exit(1)

with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 提取 repo 名称到 GitHub 链接的映射
repo_map = {url.rstrip("/").split("/")[-1]: url for url in data["github_links"]}

# 检查 repo 目录是否存在
if not os.path.exists(repo_base_path):
    print(f"Error: Path '{repo_base_path}' does not exist.")
    exit(1)

# 获取所有直接子文件夹
subfolders = [f for f in os.listdir(repo_base_path) if os.path.isdir(os.path.join(repo_base_path, f))]

# 查找空文件夹
empty_folders = [folder for folder in subfolders if not os.listdir(os.path.join(repo_base_path, folder))]

# 在 JSON 文件中查找对应的 GitHub 链接
empty_repo_links = [repo_map[folder] for folder in empty_folders if folder in repo_map]

# 生成最终 JSON 格式
result = {"github_links": empty_repo_links}

# 保存到 JSON 文件
with open(output_file_path, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=4, ensure_ascii=False)

print(f"Check complete. Found {len(empty_repo_links)} empty folders. Results saved in '{output_file_path}'.")

# 选择是否在空文件夹中创建 time_info.txt 文件
create_file = input("Do you want to create 'time_info.txt' in empty folders? (y/n): ").strip().lower()
if create_file == 'y':
    for folder in empty_folders:
        file_path = os.path.join(repo_base_path, folder, "time_info.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            pass  # 创建空文件
    print("Empty 'time_info.txt' files created in empty folders.")
else:
    print("No files were created.")
