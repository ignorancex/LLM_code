import os

base_dir = 'LLM_code/arxiv_dataset'
years = [str(year) for year in range(2020, 2026)]
seasons = ['Q1', 'Q2', 'Q3', 'Q4']

for year in years:
    for season in seasons:
        target_dir = os.path.join(base_dir, year, season)

        if not os.path.exists(target_dir):
            print(f"⚠️ 目录不存在：{target_dir}，跳过。")
            continue

        # 统计子文件夹数量
        folder_count = len([
            name for name in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, name))
        ])

        print(f"📁 目录 {target_dir} 下共有 {folder_count} 个文件夹")
