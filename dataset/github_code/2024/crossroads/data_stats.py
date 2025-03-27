import pandas as pd
import numpy as np
from tqdm import tqdm
import json

ARTIFACT_PATHS = ["corrected/country_obj_dict_adj_unfiltered.json", "corrected/country_obj_dict_no_adj_unfiltered.json", "corrected/country_obj_no_adj_tfidf_unfiltered.json"]

for path in tqdm(ARTIFACT_PATHS):
    with open(path) as f:
        country_obj_dict = json.load(f)
    print(path)
    print(len(country_obj_dict))
    # count of all values in each key
    total = 0
    for key in country_obj_dict:
        print(key, len(country_obj_dict[key]))
        total += len(country_obj_dict[key])
    print("Total artifacts : ", total)
    special = 0
    special_values = []
    if "tfidf" in path:
        for key in country_obj_dict:
            for value in country_obj_dict[key]:
                v = country_obj_dict[key][value]
                if v > 3.01 or v < 0.47:
                    special += 1
                    special_values.append((key, value))
        print("Special artifacts : ", special)
        special_dict = {}
        for key, value in special_values:
            if key not in special_dict:
                special_dict[key] = []
            special_dict[key].append(value)
        print(special_dict)
        with open("unique_associations.json", "w") as f:
            json.dump(special_dict, f, indent=4)
    print("-"*50)

"""
PATHS = ["corrected/marvl_llava.csv", "corrected/marvl_gpt.csv", "corrected/dollar_street_gpt.csv", "corrected/dollar_street_llava.csv", "corrected/dalle_street_gpt.csv", "corrected/dalle_street_llava.csv"]

for path in tqdm(PATHS):
    df = pd.read_csv(path)
    print(path)
    if "marvl" in path:
        df.rename(columns={"true_country": "true_sub_region"}, inplace=True)
    print(len(df))
    print(df.groupby("true_sub_region").count()["id"])
    print("-"*50)
    if "marvl" not in path:
        print(len(df["true_country"].unique()))
    # if "dollar" in path:
    #     print(df["true_country"].unique().tolist())
    if "dalle" in path:
        print(df["true_country"].unique().tolist())
"""
