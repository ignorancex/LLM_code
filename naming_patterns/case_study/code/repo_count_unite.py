"""
将逐季度纵向 CSV 合并为 4 份宽表：
  - cs / non_cs × (TotalFrequency / RepoCount)
"""
import csv
import os
import pandas as pd
from collections import defaultdict

# ======== 0. 常量配置 ======== #
OUT_DIR        = "LLM_code/arxiv_result/vars"          # 先前季度文件所在目录
ALL_QUARTERS   = [f"{y}Q{q}" for y in range(2025, 2026) for q in range(2, 4)]
CLS_LIST       = ["cs", "non_cs"]                      # 两个类别
OUT_FILES_META = {                                     # 输出文件名模板
    ("cs",      "freq"): "variable_TotalFrequency_cs.csv",
    ("cs",      "repo"): "variable_RepoCount_cs.csv",
    ("non_cs",  "freq"): "variable_TotalFrequency_non_cs.csv",
    ("non_cs",  "repo"): "variable_RepoCount_non_cs.csv",
}


# ======== 1. 汇总数据 ======== #
# 构造四个字典：{variable: {quarter: value}}
agg_data = {
    ("cs",     "freq"): defaultdict(dict),
    ("cs",     "repo"): defaultdict(dict),
    ("non_cs", "freq"): defaultdict(dict),
    ("non_cs", "repo"): defaultdict(dict),
}

for quarter in ALL_QUARTERS:
    for cls in CLS_LIST:
        file_path = os.path.join(OUT_DIR, f"variable_{quarter}_{cls}.csv")
        if not os.path.exists(file_path):
            # 没有该季度就留空，稍后会被 fillna(0)
            continue

        # 读入当前季度 CSV
        df = pd.read_csv(file_path)

        # 写入四个汇总字典
        for _, row in df.iterrows():
            var = row["Variable"]
            agg_data[(cls, "freq")][var][quarter] = row["TotalFrequency"]
            agg_data[(cls, "repo")][var][quarter] = row["RepoCount"]


# ======== 2. 写宽表 CSV ======== #
def dict_to_wide(data_dict: dict) -> pd.DataFrame:
    """{var: {quarter: value}} → DataFrame 并按季度列顺序排列"""
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df = df.reindex(columns=ALL_QUARTERS)   # 统一列顺序
    df.index.name = "variable"
    df.fillna(0, inplace=True)              # 缺失填 0
    return df.reset_index()

os.makedirs(OUT_DIR, exist_ok=True)

for (cls, kind), inner_dict in agg_data.items():
    wide_df = dict_to_wide(inner_dict)
    out_name = OUT_FILES_META[(cls, kind)]
    wide_df.to_csv(os.path.join(OUT_DIR, out_name), index=False)
    print(f"✅ 已写出 {out_name}")

print("\n--- 四个宽表全部生成完毕！ ---")
