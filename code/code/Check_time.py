import os

# 目标路径
year = "2024"  # 可以修改年份
repo_base_path = f"github_code/{year}"

# 检查路径是否存在
if not os.path.exists(repo_base_path):
    print(f"Error: Path '{repo_base_path}' does not exist.")
    exit(1)

# 获取所有直接子文件夹
subfolders = [f for f in os.listdir(repo_base_path) if os.path.isdir(os.path.join(repo_base_path, f))]

# 记录缺失或为空的文件夹
missing_or_empty = []

# 检查每个文件夹是否包含非空的 time_info.txt
for folder in subfolders:
    time_info_path = os.path.join(repo_base_path, folder, "time_info.txt")
    # if not os.path.exists(time_info_path) or os.path.getsize(time_info_path) == 0:
    if not os.path.exists(time_info_path):
        missing_or_empty.append(folder)

# 输出结果
if missing_or_empty:
    print("Folders missing 'time_info.txt' or containing an empty file:")
    for folder in missing_or_empty:
        print(folder)
else:
    print("All folders contain a non-empty 'time_info.txt'.")
