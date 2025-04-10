import os
import pandas as pd
from tqdm import tqdm

time_method = "by_file_as_one"

def process_file(base_name):
    base_path = f"LLM_code/output_by_quarter/{time_method}"
    years = ["2020", "2021", "2022", "2023", "2024", "2025"]
    data = {}

    for year in tqdm(years, desc=f"Processing {base_name} by year"):
        year_path = os.path.join(base_path, year)
        if not os.path.isdir(year_path):
            continue

        for quarter in os.listdir(year_path):
            quarter_path = os.path.join(year_path, quarter)
            if not os.path.isdir(quarter_path):
                continue

            file_path = os.path.join(quarter_path, f"{base_name}_{time_method}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                quarter_label = f"{year}Q{quarter[-1]}"  # 假设子文件夹命名为 Q1、Q2、Q3、Q4
                for _, row in df.iterrows():
                    name = row.iloc[0]
                    count = row.iloc[1]
                    if name not in data:
                        data[name] = {}
                    data[name][quarter_label] = data[name].get(quarter_label, 0) + count

    # 转换为 DataFrame
    merged_df = pd.DataFrame.from_dict(data, orient='index').fillna(0).astype(int).reset_index()
    merged_df = merged_df.rename(columns={"index": "Name"})

    # 添加总计列
    quarter_cols = [col for col in merged_df.columns if col != "Name"]
    merged_df["total"] = merged_df[quarter_cols].sum(axis=1)

    # 排序
    merged_df = merged_df.sort_values(by="total", ascending=False)

    return merged_df

# 处理多个 base_name
# for file in tqdm(["functions", "variables", "comments_words", "file_name_frequency"], desc="Merging files"):
for file in tqdm(["functions", "variables"], desc="Merging files"):
    result_df = process_file(file)
    output_path = f"LLM_code/output_by_quarter/{time_method}/{file}_{time_method}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"已保存季度合并结果: {output_path}")
