import os
import pandas as pd
from math import ceil

# 分别设置读取路径和保存路径
method = "by_project"
read_base_path = "normalized"
save_base_path = f"LLM_code/output_by_quarter/{method}/cumulative_counts"
os.makedirs(save_base_path, exist_ok=True)  # 如果目标路径不存在则创建

# 要处理的文件列表
file_list = [
    f"normalized_variables_{method}.csv"
]

# 设置分割份数
num_parts = 3  # ✅ 可以改为任意整数

for filename in file_list:
    read_path = os.path.join(read_base_path, filename)

    try:
        df = pd.read_csv(read_path)
        total_rows = len(df)
        chunk_size = ceil(total_rows / num_parts)

        base_name = filename.replace(".csv", "")

        for i in range(num_parts):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)

            df_chunk = df.iloc[start_idx:end_idx]

            save_path = os.path.join(save_base_path, f"{base_name}_part{i+1}.csv")
            df_chunk.to_csv(save_path, index=False)

            print(f"✅ 已保存分片: {save_path}（行数: {len(df_chunk)}）")

    except Exception as e:
        print(f"❌ 处理文件 {filename} 出错：{e}")
