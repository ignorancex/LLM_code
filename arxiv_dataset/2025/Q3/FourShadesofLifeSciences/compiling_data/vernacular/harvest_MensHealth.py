#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "transfer url data from Men's Health https://www.menshealth.com webpage with information and adding to data directory"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import argpars

def get_infos(tags, url):
    if url.startswith("https://www.menshealth.com"):

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            # get article text
            try:
                text = soup.get_text()
                text = text.replace("\n", " ").replace("\r", " ").replace("→", " ")

                try:
                    cleaned_text = re.split(r"Advertisement - Continue Reading", text)[
                        0
                    ].strip()
                except Exception as e:
                    print(e)
                try:
                    article_title = get_article_title(url)
                except Exception as e:
                    print(e)
                try:
                    date_pattern = r"Published: (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}Save Article"
                except Exception as e:
                    print(e)
                try:
                    cleaned_text = re.split(date_pattern, cleaned_text)[1].strip()
                except Exception as e:
                    print(e)
                try:
                    cleaned_text = cleaned_text.replace("Getty Images", "")
                except Exception as e:
                    print(e)

                infos = pd.DataFrame(
                    {
                        "category_id": "popular",
                        "text_id": "MensHealth" + article_title,
                        "tags": tags,
                        "venue": "",
                        "data-source": "MensHealth",
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
    article_title = url.split("https://www.menshealth.com/")[1]
    return article_title


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    urls_df = pd.read_csv(args.input
    )
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
        tags = str(row[0])
        print(tags)
        url = str(row[1])
        print(url)

        infos = get_infos(tags, url)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

    infos_df.to_csv(args.output,
        index=False,
    )
    print("done")


if __name__ == "__main__":
    main()
