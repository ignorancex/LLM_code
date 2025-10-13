#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "harvest data from https://www.infowars.com/category/4/ health category and add it to data directory"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import argparse

def get_infos(url):
    time.sleep(7)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # get article text
    try:
        text = soup.get_text()

    except Exception as e:
        print(e)

    try:
        text = text.replace("\n", "").replace("\r", "")
    except Exception as e:
        print(e)

    article_title = get_article_title(url)

    infos = pd.DataFrame(
        {
            "category_id": "disinfo",
            "text_id": "InfoWars:" + str(article_title),
            "venue": "",
            "data-source": "InfoWars",
            "url": [url],
            "tags": "",
            "text": [text],
        }
    )

    return infos


def get_article_title(url):
    try:
        article_title = url.split("https://www.infowars.com/posts/")[1]
        return article_title
    except Exception as e:
        print(e)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    urls_df = pd.read_csv(args.input)
    urls_df.drop_duplicates(inplace=True)
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
    for index, row in urls_df.iterrows():
        print(row[0])
        url = row[0]
        infos = get_infos(url)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

        infos_df.to_csv(args.output,
            index=False,
        )
    print("done")


if __name__ == "__main__":
    main()
