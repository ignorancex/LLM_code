# import os
# import pandas as pd
#
#
# def process_file(file_name):
#     base_path = "output_project"
#     years = ["2020", "2021", "2022", "2023", "2024", "2025"]
#     data = {}
#
#     for year in years:
#         file_path = os.path.join(base_path, year, file_name)
#         if os.path.exists(file_path):
#             df = pd.read_csv(file_path)
#             for _, row in df.iterrows():
#                 name = row.iloc[0]
#                 count = row.iloc[1]
#                 if name not in data:
#                     data[name] = {y: 0 for y in years}
#                 data[name][year] = count
#
#     # 转换为 DataFrame
#     merged_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
#     merged_df.columns = ["Name"] + years
#
#     # 计算总数
#     merged_df["total"] = merged_df[years].sum(axis=1)
#
#     # 计算增长率 (2024年相对2020年增长率)
#     merged_df["rate(2020->2024)"] = merged_df.apply(
#         lambda row: (row["2024"] - row["2020"]) / row["2020"] if row["2020"] != 0 else "inf", axis=1)
#
#     # 按总数排序
#     merged_df = merged_df.sort_values(by="total", ascending=False)
#
#     return merged_df
#
#
# # 处理四个不同的 CSV 文件
# for file in ["functions.csv", "variables.csv", "comments_words.csv", "file_name_frequency.csv"]:
#     result_df = process_file(file)
#     result_df.to_csv(f"data_byproject_{file}", index=False)
#     print(f"合并完成: merged_{file}")


import os
import pandas as pd
import re


def process_file(base_name):
    base_path = "output_file"
    years = ["2020", "2021", "2022", "2023", "2024", "2025"]
    data = {}

    for year in years:
        file_path = os.path.join(base_path, year, f"{base_name}_{year}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                name = row.iloc[0]
                count = row.iloc[1]
                if name not in data:
                    data[name] = {y: 0 for y in years}
                data[name][year] = count

    # 转换为 DataFrame
    merged_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    merged_df.columns = ["Name"] + years

    # 计算总数
    merged_df["total"] = merged_df[years].sum(axis=1)

    # 计算增长率 (2024年相对2020年增长率)
    merged_df["rate(2020->2024)"] = merged_df.apply(
        lambda row: (row["2024"] - row["2020"]) / row["2020"] if row["2020"] != 0 else None, axis=1)

    # 按总数排序
    merged_df = merged_df.sort_values(by="total", ascending=False)

    return merged_df


# 处理四个不同的 CSV 文件
for file in ["functions", "variables", "comments_words", "file_name_frequency"]:
    result_df = process_file(file)
    result_df.to_csv(f"data_byfile_{file}.csv", index=False)
    print(f"合并完成: merged_{file}.csv")