#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "calculation of average text length"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import pandas as pd
import argparse


def calculate_avarage_text_length(lengths_id, i, df_id):
    for index, row in df_id.iterrows():
        try:
            length = len(row["text"])
            lengths_id += length
            # print(lengths_id)
            i += 1
        except Exception as e:
            print(e)

    average_length = lengths_id / i

    return average_length, i


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    args = argparser.parse_args()
    df = pd.read_csv(args.input)

    ids = ["scientific", "popular", "alternative_science", "disinfo"]

    for id in ids:
        lengths_id = 0
        i = 0
        df_id = df.loc[df["category_id"] == f"{id}"]
        average_length, i = calculate_avarage_text_length(lengths_id, i, df_id)
        print(f"average text length of {id}:", average_length)
        print(f"number:", i)


if __name__ == "__main__":
    main()
