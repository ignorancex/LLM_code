#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "harvest data from https://natural.news and add it to data directory"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import argparse

def get_infos(row, i):
    print(row)
    url = row
    i = +1

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # get article text
    try:
        text = soup.get_text()

    except Exception as e:
        print(e)

    try:
        text = (
            text.replace("\n", "")
            .replace("\r", "")
            .replace(
                "Your NameYour emailMessage or CancelSCIENCEFOODHEALTHMEDICINEPOLLUTIONCANCERCLIMATE",
                "",
            )
        )
        # print(text)
    except Exception as e:
        print(e)

    try:
        clean_text = text.split("RECENT NEWS & ARTICLES")[0]
    except Exception as e:
        print(e)
    try:
        cleaner_text = clean_text.split("Sources include:")[0]
    except Exception as e:
        print(e)

    try:
        txt = re.sub("[^a-zA-Z0-9 \n\.]", "", cleaner_text)
    except Exception as e:
        print(e)

    article_title = get_article_title(url)

    infos = pd.DataFrame(
        {
            "category_id": "disinfo",
            "text_id": "HealthNews:" + str(article_title),
            "venue": "",
            "data_source": "HealthNews",
            "url": [url],
            "tags": "",
            "text": [text],
        }
    )

    return infos, i


def get_article_title(url):
    try:
        article_title = url.split("https://www.health.news")[1]
        return article_title
    except Exception as e:
        print(e)
    try:
        article_title = url.split("https://www.climate.news")[1]
        return article_title
    except Exception as e:
        print(e)
    try:
        article_title = url.split("https://www.censoredscience.news")[1]
        return article_title
    except Exception as e:
        print(e)
    try:
        article_title = url.split("https://www.medicine.news")[1]
        return article_title
    except Exception as e:
        print(e)
    try:
        article_title = url.split("https://www.pollution.news")[1]
        return article_title
    except Exception as e:
        print(e)
    try:
        article_title = url.split("https://www.cancer.news")[1]
        return article_title
    except Exception as e:
        print(e)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    urls_df = pd.read_csv(args.input, sep=",")
    urls_df.drop_duplicates(inplace=True)

    urls_df["deduplicate"] = urls_df["url"].replace(regex="https?://\S+/", value="")
    urls_df["deduplicate"] = urls_df["deduplicate"].replace(
        regex=["((?:19|20)\\d\\d)-(0?[1-9]|1[012])-([12][0-9]|3[01]|0?[1-9])-"],
        value="",
    )

    urls_df = urls_df.drop_duplicates(subset=["deduplicate"])

    i = 0
    infos_df = pd.DataFrame(
        columns=[
            "category_id",
            "text_id",
            "tags",
            "venue",
            "data_source",
            "url",
            "text",
        ]
    )
    for index, row in urls_df.iterrows():
        row = str(row[0])
        infos, i = get_infos(row, i)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

        infos_df.to_csv(argsg.output, index=False)
    print(i)
    print("done")


if __name__ == "__main__":
    main()
