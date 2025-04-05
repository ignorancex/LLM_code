import os
import shutil

# 年份与季度
year = '2025'
seasons = ['Q1']

for season in seasons:
    src_dir = f'github_code/{year}/{season}'
    dst_dir = f'LLM_code/arxiv_dataset/{year}/{season}'

    # 如果源目录不存在，跳过
    if not os.path.exists(src_dir):
        print(f"⚠️ 源目录不存在：{src_dir}，跳过该季度。")
        continue

    # 创建目标目录（如果不存在）
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历源目录中的所有文件和文件夹
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        # 移动文件或文件夹
        shutil.move(src_path, dst_path)

    print(f"✅ {season} 所有内容已成功移动。")

print("🎉 全部季度处理完毕。")
