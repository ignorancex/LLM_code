#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "transfer url data from MedlinePlus webpage with patient information and adding to data directory"
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


def get_infos(url, tags):
    if url.startswith("https://magazine.medlineplus.gov"):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # get article text
            try:
                text = soup.get_text()
                text = text.replace("\n", " ").replace("\r", " ").replace("→", " ")
                text = re.sub("[^a-zA-Z0-9 \n\.]", "", text)
                article_title = get_article_title(url)
                cleaned_text = re.split(r"Alternative accessible version", text)[
                    0
                ].strip()

                try:
                    cleaned_text = re.split(r"UpdatesSearchSearch", cleaned_text)[
                        1
                    ].strip()
                except Exception as e:
                    print(e)
                print(cleaned_text)
                try:
                    cleaned_text = re.split(r"You May Also Like", cleaned_text)[
                        0
                    ].strip()
                except Exception as e:
                    print(e)
                # cleaned_text = re.split(r'Find Out More', cleaned_text)[1].strip()
                try:
                    cleaned_text = re.split(r"Find Out More", cleaned_text)[0].strip()
                except Exception as e:
                    print(e)

                infos = pd.DataFrame(
                    {
                        "category_id": "popular",
                        "text_id": "MedlinePlus" + article_title,
                        "tags": tags,
                        "venue": "",
                        "data-source": "MedlinePlus",
                        "url": [url],
                        "text": [cleaned_text],
                    }
                )
                return infos

            except Exception as e:
                print(e)
        except Exception as e:
            print(e)


def get_article_title(url):
    article_title = url.split("https://magazine.medlineplus.gov")[1]
    article_title = article_title.split(".html")[0]
    return article_title


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
        url = str(row[1])
        tags = str(row[0])

        infos = get_infos(url, tags)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

    infos_df.to_csv(args.output,
        index=False,
    )
    print("done")


if __name__ == "__main__":
    main()
