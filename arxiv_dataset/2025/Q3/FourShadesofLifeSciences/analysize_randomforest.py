#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "analyze decision tree rules from  random forrest"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import glob
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
args = argparser.parse_args()
argparser.add_argument(args.input_folder, help="Path to folder containing random forests")


rules_list = []
df_all = pd.DataFrame()

for name in glob.glob(args.input_folder):
    df = pd.read_csv(name)
    df.columns = ["term"]
    df = df.stack().str.replace(r"|---", "").unstack()
    df = df.stack().str.replace(r"|   |", "").unstack()
    df = df.stack().str.replace(r"|", "").unstack()
    df = df.stack().str.replace("<=", "").unstack()
    df = df.stack().str.replace(">", "").unstack()
    df = df.stack().str.replace("truncated branch of depth", "").unstack()
    df_all = pd.concat([df_all, df])

    items = df.iloc[:, 0].tolist()


    for term in items:
        cleaned_data = term.strip()

        cleaned_data = cleaned_data.split(" ")[0]
        cleaned_data = cleaned_data.replace("class:", "").strip()
        cleaned_data = cleaned_data.replace("\n", "")
        rules_list.append(cleaned_data)


# Initialize counts as a dictionary
counts = {}

# Iterate over each term in the list
for term in rules_list:
    if term in counts:
        counts[term] += 1
    else:
        counts[term] = 1
counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
print(counts)
