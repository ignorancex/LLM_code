#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = (
    "retrieve text from PDF-URLs BMC Complementary Medicine and Therapies"
)
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import requests
import pandas as pd
import re
from papermage.recipes import CoreRecipe
import argparse


def download_pdf(url, i):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    }

    # Get response object for link
    response = requests.get(url, headers=headers)
    # Write content in pdf file dummy
    path = "dummy_PDF.pdf"
    pdf = open(path, "wb")
    pdf.write(response.content)
    pdf.close()

    return path


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


def clean_text(pdf_txt):
    cleaned_txt = " ".join(pdf_txt.split())
    cleaned_txt = re.sub("[^a-zA-Z0-9 \n\.]", " ", cleaned_txt)
    cleaned_txt = re.sub(
        r"Annotated Entity\s+ID\s+\d+\s+Spans\s+True\s+Boxes\s+True\s+Text",
        "",
        cleaned_txt,
    )

    return cleaned_txt


def compile_infos(pdf_txt, df, text_id, url, tag):
    row = pd.DataFrame(
        {
            "category_id": "alternative_science",
            "text_id": text_id,
            "tags": tag,
            "venue": "",
            "data-source": "CompMedTherapies",
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
    urls_df = pd.read_csv(args.input)
    i = 0

    # initiate df with information columns
    df = pd.DataFrame(
        columns=[
            "category-id",
            "text_id",
            "tags",
            "venue",
            "data-source",
            "url",
            "text",
        ]
    )
    # loop through each document-url
    for index, row in urls_df.iterrows():
        i += 1
        print(i)

        url = str(row[1])

        text_id = url.split("pdf/")[1]
        tag = str(row[0])

        # download pdf in dummy
        path = download_pdf(url, i)

        # parse pdf to string
        pdf_txt = pdf_to_text(path)
        if pdf_txt is None:
            continue

        # preprocess string
        cleaned_txt = clean_text(pdf_txt)

        # compile information  df
        df = compile_infos(cleaned_txt, df, text_id, url, tag)

    df.to_csv(args.output,
              mode="a",
        index=False,
        header=False,
    )
    print("done")


if __name__ == "__main__":
    main()
