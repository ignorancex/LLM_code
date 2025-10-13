#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = (
    "retrieve text from PDF-URLs "
    "Journal of Evidence-Based Integrative Medicine / JEBIM"
)
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import requests
import pandas as pd
import re
import glob
from papermage.recipes import CoreRecipe
import time
import argparse


def download_pdf(url):
    time.sleep(1)
    path = "data/2025-01-28_dummy_anthropo-PDF.pdf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
            print("Downloaded")
            return path
    else:
        print(f"Failed to download PDF. HTTP Status Code: {response.status_code}")


def pdf_to_text(path):
    recipe = CoreRecipe()
    try:
        doc = recipe.run(path)
        txt = []
        doc = doc.symbols
        txt.append(doc)

        pdf_text = "".join(str(txt))
        return pdf_text
    except:
        print("WARNING NO PDF")
        return None


def clean_text(pdf_text):
    cleaned_txt = " ".join(pdf_text.split())

    try:
        cleaned_txt = re.sub("[^a-zA-Z0-9 \n\.]", "", cleaned_txt)
    except Exception as e:
        print(e)

    return cleaned_txt


def compile_infos(pdf_txt, df, text_id, url, tag):
    row = pd.DataFrame(
        {
            "category_id": "alternative_science",
            "text_id": text_id,
            "tags": tag,
            "venue": "",
            "data-source": "JEBIM",
            "url": url,
            "text": pdf_txt,
        },
        index=[0],
    )
    df = pd.concat([df, row], ignore_index=True)
    return df


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    # read csv with URLS
    urls_df = pd.read_csv(args.input)


    # initiate df with information columns
    df = pd.DataFrame(
        columns=[
            "category_id",
            "text_id",
            "tags",
            "venue",
            "data-source",
            "url",
            "text",
        ])
    i = 0
    # loop through each document-url
    for index, row in urls_df.iterrows():
        i += 1
        print("Processing file:", i, "URL", row["url"], "with tag", row['topic'])

        # download pdf in dummy
        url = row["url"]
        tag = row['topic']
        text_id = url

        path = download_pdf(url, i)

        # parse pdf to string
        pdf_txt = pdf_to_text(path)

        if pdf_txt is None:
            continue


        cleaned_txt = clean_text(pdf_txt)

        df = compile_infos(cleaned_txt, df, text_id, url, tag)

    df.to_csv(args.output,
        mode="a",
        index=False,
        header=False,
    )
    print("done")


if __name__ == "__main__":
    main()
