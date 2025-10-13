#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "get data from desinfo webpage Mercolas Censored library and adding to data directory: https://www.mercola.com/ e.g.: https://takecontrol.substack.com/p/what-happens-during-menopause"

__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import sys
import argparse


def get_infos(row):
    if row.startswith("https://takecontrol.substack.com/p/"):
        url = "".join(row)
        try:
            response = requests.get(url)

            soup = BeautifulSoup(response.text, "html.parser")

            # get article text
            try:
                text = soup.get_text()
                text = text.replace("\n\n", " ").replace("\r", " ").replace("→", " ")
                text = text.strip()
                clean_text = text.split("Sources include:", 1)

                # cleaned_text = re.sub('[^a-zA-Z0-9 \n\.]', '', clean_text)
                # print(clean_text)
                article_title = get_article_title(url)

                infos = pd.DataFrame(
                    {
                        "category_id": "disinfo",
                        "text_id": "Mercola:" + article_title,
                        "venue": "",
                        "data-source": "Mercola",
                        "url": [url],
                        "tags": "",
                        "text": [clean_text],
                    }
                )
                return infos

            except Exception as e:
                print(e)
        except Exception as e:
            print(e)


def get_article_title(url):
    article_title = url.split("https://takecontrol.substack.com/p/")[1]
    return article_title


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    urls_df = pd.read_csv(args.input, sep=",")
    urls_df.drop_duplicates(inplace=True)

    infos_df = pd.DataFrame(
        columns=[
            "category_id",
            "text_id",
            "venue",
            "data_source",
            "url",
            "tags",
            "text",
        ]
    )
    for index, row in urls_df.iterrows():
        row = str(row[0])

        infos = get_infos(row)
        print(infos)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

    infos_df.to_csv(args.output, index=False)
    print("done")


if __name__ == "__main__":
    main()
