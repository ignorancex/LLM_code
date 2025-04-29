import os
import shutil

# 定义年份和季度范围
years = [str(y) for y in range(2020, 2026)]
seasons = ["Q1", "Q2", "Q3", "Q4"]

# 记录所有缺失或为空的文件夹
all_missing_or_empty = []

for year in years:
    for season in seasons:
        repo_base_path = f"LLM_code/arxiv_dataset_cpp/{year}/{season}"

        # 检查路径是否存在
        if not os.path.exists(repo_base_path):
            print(f"Warning: Path '{repo_base_path}' does not exist.")
            continue

        # 获取所有直接子文件夹
        subfolders = [f for f in os.listdir(repo_base_path) if os.path.isdir(os.path.join(repo_base_path, f))]

        # 检查每个文件夹是否包含 time_info.txt 且文件不为空
        for folder in subfolders:
            time_info_path = os.path.join(repo_base_path, folder, "time_info_cpp.txt")
            if not os.path.exists(time_info_path) or os.path.getsize(time_info_path) == 0:
                all_missing_or_empty.append(os.path.join(repo_base_path, folder))

# 输出结果
if all_missing_or_empty:
    print("\nFolders missing 'time_info.txt' or containing an empty file:")
    for folder in all_missing_or_empty:
        print(folder)

    # 询问用户是否删除
    choice = input("\nDo you want to delete these folders? (y/n): ").strip().lower()
    if choice == 'y':
        for folder in all_missing_or_empty:
            shutil.rmtree(folder)
            print(f"Deleted: {folder}")
    else:
        print("No folders were deleted.")
else:
    print("All folders contain a non-empty 'time_info.txt'.")
