import csv
import os
import pandas as pd
from collections import defaultdict

OUT_DIR        = "LLM_code/arxiv_result/vars"          
ALL_QUARTERS   = [f"{y}Q{q}" for y in range(2025, 2026) for q in range(2, 4)]
CLS_LIST       = ["cs", "non_cs"]                     
OUT_FILES_META = {                                     
    ("cs",      "freq"): "variable_TotalFrequency_cs.csv",
    ("cs",      "repo"): "variable_RepoCount_cs.csv",
    ("non_cs",  "freq"): "variable_TotalFrequency_non_cs.csv",
    ("non_cs",  "repo"): "variable_RepoCount_non_cs.csv",
}


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
            continue

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            var = row["Variable"]
            agg_data[(cls, "freq")][var][quarter] = row["TotalFrequency"]
            agg_data[(cls, "repo")][var][quarter] = row["RepoCount"]


def dict_to_wide(data_dict: dict) -> pd.DataFrame:

    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df = df.reindex(columns=ALL_QUARTERS)   
    df.index.name = "variable"
    df.fillna(0, inplace=True)              
    return df.reset_index()

os.makedirs(OUT_DIR, exist_ok=True)

for (cls, kind), inner_dict in agg_data.items():
    wide_df = dict_to_wide(inner_dict)
    out_name = OUT_FILES_META[(cls, kind)]
    wide_df.to_csv(os.path.join(OUT_DIR, out_name), index=False)

