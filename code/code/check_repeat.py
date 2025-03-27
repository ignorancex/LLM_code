import os
year = '2024'
src_dir = f'github_code/{year}'
dst_dir = f'LLM_code/dataset/github_code/{year}'

# 统计同名文件夹数量
conflict_count = 0
conflict_list = []

for item in os.listdir(src_dir):
    src_path = os.path.join(src_dir, item)
    dst_path = os.path.join(dst_dir, item)
    
    # 如果是文件夹且目标中也存在同名文件夹
    if os.path.isdir(src_path) and os.path.isdir(dst_path):
        conflict_count += 1
        conflict_list.append(item)

print(f"⚠️ 共有 {conflict_count} 个同名文件夹：")
for name in conflict_list:
    print(f" - {name}")
