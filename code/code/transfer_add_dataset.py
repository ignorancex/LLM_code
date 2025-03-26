import os
import shutil

# 原始路径和目标路径
src_dir = 'github_code/2020'
dst_dir = 'LLM_code/dataset/github_code/2020'

# 创建目标目录（如果不存在）
os.makedirs(dst_dir, exist_ok=True)

# 遍历源目录中的所有文件和文件夹
for item in os.listdir(src_dir):
    src_path = os.path.join(src_dir, item)
    dst_path = os.path.join(dst_dir, item)
    
    # 移动文件或文件夹
    shutil.move(src_path, dst_path)

print("✅ 所有内容已成功移动。")
