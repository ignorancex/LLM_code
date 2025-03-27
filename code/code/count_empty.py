import os

def count_folders_with_only_time_info(root_dir):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if len(filenames) == 1 and filenames[0] == 'time_info.txt' and not dirnames:
            count += 1
            # print(f"Matched folder: {dirpath}")
    print(f"\nTotal folders containing only 'time_info.txt': {count}")

# 设置路径
root_directory = 'LLM_code/dataset/github_code/2025'
count_folders_with_only_time_info(root_directory)
