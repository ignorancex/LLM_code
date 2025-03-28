import os

target_dir = 'LLM_code/arxiv_dataset/2025/Q1'

# 统计该目录下的所有子文件夹
folder_count = len([
    name for name in os.listdir(target_dir)
    if os.path.isdir(os.path.join(target_dir, name))
])

print(f"📁 目录 {target_dir} 下共有 {folder_count} 个文件夹")
