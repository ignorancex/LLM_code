#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "harvest data from https://www.mayoclinic.org/drugs-supplements webpage with information on drugs and supplements"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_infos(row):
    if row.startswith("https://www.mayoclinic.org/drugs-supplements/"):
        url = "".join(row)
        try:
            hdr = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=hdr)
            print(response)
            soup = BeautifulSoup(response.text, "html.parser")

            # get article text
            try:
                text = soup.get_text()
                text = text.replace("\n", "").replace("\r", "").replace("→", "")
                text = re.sub("[^a-zA-Z0-9 \n\.]", "", text)
                article_title = get_article_title(url)
                text = " ".join(text.split())
                infos = pd.DataFrame(
                    {
                        "category_id": "popular_science",
                        "text_id": "MayoClinic_" + article_title,
                        "venue": "",
                        "data-source": "MayoClinic",
                        "url": [url],
                        "tags": "",
                        "text": [text],
                    }
                )
                return infos

            except Exception as e:
                print(e)
        except Exception as e:
            print(e)


def get_article_title(url):
    article_title = url.split(
        "https://www.mayoclinic.org/drugs-supplements/drugs-supplements-"
    )[1]
    article_title = article_title.split("/art-")[0]
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
            "venue",
            "data-source",
            "url",
            "tags",
            "text",
        ]
    )
    for index, row in urls_df.iterrows():
        row = str(row[0])

        infos = get_infos(row)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

    infos_df.to_csv(args.output, index=False)
    print("done")


if __name__ == "__main__":
    main()
