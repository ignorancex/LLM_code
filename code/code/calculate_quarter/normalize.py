import os
import pandas as pd
from tqdm import tqdm

time_method = "by_project"
input_dir = f"LLM_code/output_by_quarter/{time_method}"
files = ["functions", "variables", "comments_words", "file_name"]

for file in tqdm(files, desc="Normalizing and computing growth rate"):
    input_path = os.path.join(input_dir, f"{file}_{time_method}.csv")
    df = pd.read_csv(input_path)

    # 找出所有季度列（排除 Name 和 total）
    quarter_cols = [col for col in df.columns if col not in ["Name", "total"]]
    quarter_cols_sorted = sorted(quarter_cols)  # 按时间排序（2020Q1, 2020Q2, ...）

    # 计算每个季度的总数
    col_sums = df[quarter_cols_sorted].sum()

    # 创建归一化 DataFrame
    normalized_df = df.copy()
    for col in quarter_cols_sorted:
        if col_sums[col] != 0:
            normalized_df[col] = normalized_df[col] / col_sums[col]
        else:
            normalized_df[col] = 0

    # 保留 total 列
    normalized_df["total"] = df["total"]

    # 计算增长率
    if len(quarter_cols_sorted) >= 8:
        first4 = quarter_cols_sorted[:4]
        last4 = quarter_cols_sorted[-4:]

        # 均值
        first4_mean = normalized_df[first4].mean(axis=1)
        last4_mean = normalized_df[last4].mean(axis=1)

        # 增长率计算（避免除以0）
        growth_rate = (last4_mean - first4_mean) / first4_mean.replace(0, float('nan'))
        growth_rate = growth_rate.fillna(0)  # 将 NaN（因0除）转为0
        normalized_df["growth_rate"] = growth_rate
    else:
        print(f"季度数量不足8，跳过增长率计算：{file}")
        normalized_df["growth_rate"] = 0

    # 输出新文件
    output_path = os.path.join(input_dir, f"normalized_{file}_{time_method}.csv")
    normalized_df.to_csv(output_path, index=False)
    print(f"完成归一化与增长率计算：{output_path}")
