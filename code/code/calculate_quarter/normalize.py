import os
import re
import pandas as pd
from tqdm import tqdm

# 将"YYYYQ#"格式的列名解析为(YYYY, #)
def parse_quarter(col_name: str):
    match = re.match(r"(\d{4})Q([1-4])$", col_name.strip())  # 确保匹配严格季度格式
    if match:
        return int(match.group(1)), int(match.group(2))
    return (9999, 9999)  # 不符合的放到最后

time_method = "by_pub"
input_dir = f"LLM_code/output_by_quarter/{time_method}/raw_data"
output_dir = f"LLM_code/output_by_quarter/{time_method}/normalization"
os.makedirs(output_dir, exist_ok=True)

# files = ["functions", "variables", "file_name", "comment_words"]
files = ["nonzero_comment_words"]

for file in tqdm(files, desc="Normalizing and computing growth rate"):
    input_path = os.path.join(input_dir, f"{file}_{time_method}.csv")
    df = pd.read_csv(input_path)

    # 清理列名中的空格与特殊字符
    df.columns = df.columns.str.strip()

    # 获取所有季度列（排除非季度列）
    quarter_cols = [col for col in df.columns if re.match(r"\d{4}Q[1-4]$", col)]
    quarter_cols_sorted = sorted(quarter_cols, key=parse_quarter)

    # 打印排序后的列（用于调试）
    # print(f"[{file}] 排序后的季度列：{quarter_cols_sorted}")

    # 计算每个季度的总数
    col_sums = df[quarter_cols_sorted].sum()

    # 归一化处理
    normalized_df = df.copy()
    for col in quarter_cols_sorted:
        normalized_df[col] = normalized_df[col] / col_sums[col] if col_sums[col] != 0 else 0

    # 保留 total 列和 Name 列（如有）
    if "total" in df.columns:
        normalized_df["total"] = df["total"]
    if "Name" in df.columns:
        normalized_df["Name"] = df["Name"]

    # 计算增长率
    if len(quarter_cols_sorted) >= 8:
        first4 = quarter_cols_sorted[:4]
        last4 = quarter_cols_sorted[-4:]
        first4_mean = normalized_df[first4].mean(axis=1)
        last4_mean = normalized_df[last4].mean(axis=1)
        growth_rate = (last4_mean - first4_mean) / first4_mean.replace(0, float('nan'))
        normalized_df["growth_rate"] = growth_rate.fillna(0)
    else:
        print(f"季度数量不足8，跳过增长率计算：{file}")
        normalized_df["growth_rate"] = 0

    # 重新排列列顺序：Name + sorted_quarters + total + growth_rate
    columns_order = []
    if "Name" in normalized_df.columns:
        columns_order.append("Name")
    columns_order += quarter_cols_sorted
    if "total" in normalized_df.columns:
        columns_order.append("total")
    columns_order.append("growth_rate")

    normalized_df = normalized_df[columns_order]

    # 写入文件
    output_path = os.path.join(output_dir, f"normalized_{file}_{time_method}.csv")
    normalized_df.to_csv(output_path, index=False)
    print(f"完成归一化与增长率计算：{output_path}")
