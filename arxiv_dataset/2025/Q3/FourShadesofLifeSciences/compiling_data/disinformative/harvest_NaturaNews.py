#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = (
    "harvest data from https://www.naturalnews.com and add it to data directory"
)
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import argparse

def get_infos(i, row):
    tag = row[0]
    url = row[1]
    i = +1

    article_title = get_article_title(url, i)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # get article text
    try:
        text = soup.get_text()

    except Exception as e:
        print(e)

    cleaned_txt = " ".join(text.split())
    cleaned_txt = re.sub("[^a-zA-Z0-9 \n\.]", " ", cleaned_txt)

    infos = pd.DataFrame(
        {
            "category_id": "disinfo",
            "text_id": "NaturalNews:" + str(article_title),
            "venue": "",
            "data_source": "NaturalNews",
            "url": [url],
            "tags": [tag],
            "text": [cleaned_txt],
        }
    )

    return infos, i


def get_article_title(url, i):
    print(i)
    print(url)
    try:
        article_title = url.split("https://www.naturalnews.com")[1]
        return article_title
    except Exception as e:
        print(e)

    return article_title


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    urls_df = pd.read_csv(args.input)
    urls_df.drop_duplicates(inplace=True)

    i = 0
    infos_df = pd.DataFrame(
        columns=[
            "category_id",
            "text_id",
            "tags",
            "venue",
            "data-source",
            "url",
            "text",
        ]
    )
    for i, row in urls_df.iterrows():
        infos, i = get_infos(i, row)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

    infos_df.to_csv(args.output,
        index=False,
    )
    print("done")


if __name__ == "__main__":
    main()
