#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = (
    "harvest data from  https://healthimpactnews.com and add it to data directory"
)
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
    tag = row[0]
    url = row[1]
    i = +1

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # get article text
    try:
        text = soup.get_text()

    except Exception as e:
        print(e)

    try:
        text = text.split("Having problems receiving our emails?")[0]
        text = text.split("See Also:")[0]
        text = text.split("Print This Post", 1)[1]
        text = text.replace("Print this Post", "").replace("Subscribe", "").replace()
        text = text.replace(
            "Categories Coconut Health Real Food Nutrition Medicine Watch Alternative Health Traditional Agriculture",
            "",
        )

    except Exception as e:
        print(e)

    try:
        txt = re.sub("[^a-zA-Z0-9 \n\.]", "", text)
    except Exception as e:
        print(e)
    clean = txt.replace("\n", "").replace("\r", "")
    print(clean)

    article_title = get_article_title(url)

    infos = pd.DataFrame(
        {
            "category_id": "disinfo",
            "text_id": "HealthImpactNews:" + str(article_title),
            "tags": tag,
            "venue": "",
            "data_source": "HealthImpactNews",
            "url": [url],
            "text": [clean],
        }
    )

    return infos, i


def get_article_title(url):
    try:
        article_title = url.split("https://healthimpactnews.com/2023/")[1]
        return article_title
    except Exception as e:
        print(e)

    try:
        article_title = url.split("https://healthimpactnews.com/2022/")[1]
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
    print(urls_df.tail())
    print("nachher", urls_df.shape)
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
        infos, i = get_infos(row, i)
        infos_df = pd.concat([infos_df, infos], ignore_index=True)

        infos_df.to_csv(args.output,
            index=False,
        )
    print(i)
    print("done")


if __name__ == "__main__":
    main()
