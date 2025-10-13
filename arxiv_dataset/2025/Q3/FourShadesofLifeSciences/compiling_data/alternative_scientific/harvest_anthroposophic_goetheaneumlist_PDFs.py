#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "retrieve text from URLs file urls_anthroposophic_goetheaneum.csv. URLs mentioned in Physicians' Association for Anthroposophic Medicine (PAAM, https://anthroposophicmedicine.org/) as well as in: literature lists 2017-2020 Anthroposophic Medicine School of Spiritual Science Medical Section at the Goetheanum (https://medsektion-goetheanum.org/en/research/publications/journal-contributions-on-research-in-anthroposophic-medicine-2017-2019)"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import pdftotext
import requests
import pandas as pd
import re
from papermage.recipes import CoreRecipe
import argparse


def download_pdf(url, i):
    # Get response object for link
    response = requests.get(url)

    # Write content in pdf file dummy
    path = "data/dummy_anthropo-PDF.pdf"
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
        pdf_text = ''.join(str(txt))
        return pdf_text

    except:
        print("WARNING NO PDF")
        return None


def clean_text(pdf_text):
    cleaned_txt = " ".join(pdf_text.split())
    cleaned_txt = re.sub("[^a-zA-Z0-9 \n\.]", "", cleaned_txt)
    print(cleaned_txt)

    return cleaned_txt


def compile_infos(pdf_txt, df, text_id, url, i):
    row = pd.DataFrame(
        {
            "category_id": "alternative_science",
            "text_id": text_id,
            "venue": "",
            "data-source": "PAAM/Goetheaneum-list",
            "url": url,
            "tags": "",
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


    # initiate index
    i = 0

    # initiate df with information columns
    df = pd.DataFrame(
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

    # loop through each document-url
    for index, row in urls_df.iterrows():
        i += 1
        print("Processing file:", i, "DOI", row["doi"])

        # download pdf in dummy
        url = row["url"]
        path = download_pdf(url, i)

        # parse pdf to string
        pdf_txt = pdf_to_text(path)

        if pdf_txt is None:
            continue

        # preprocess string
        print("now cleaning")
        cleaned_text = clean_text(pdf_txt)

        # get doi
        text_id = row["doi"]

        # compile information  df
        df = compile_infos(cleaned_text, df, text_id, url, i)


    df.to_csv(args.output,
        mode="a",
        index=False,
        header=False,
    )
    print("done")


if __name__ == "__main__":
    main()
