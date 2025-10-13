#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "harvest articles from https://www.ijrh.org"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import requests
from bs4 import BeautifulSoup
import pandas as pd
from pypdf import PdfReader
import PyPDF2
from papermage.recipes import CoreRecipe
import time
import argparse


def get_urls(url):
    # Requests URL and get response object

    response = requests.get(url)
    # Parse text obtained
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all hyperlinks present on webpage
    links = soup.find_all("a")

    complete_links = []
    for link in links:
        if "cgi/viewcontent" in str(link):

            # get urls
            complete_link = str(link).split('href="', -1)[1].split('" target=')[0]
            complete_links.append(complete_link)
    return complete_links


def download_pdf(complete_link):
    # Get response object for link
    time.sleep(3)
    path = "data/PDF_dummy.pdf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    response = requests.get(complete_link, headers=headers)

    # Write content in pdf file
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
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


def clean_text(pdf_txt):
    cleaned_txt = " ".join(pdf_txt.split())
    return cleaned_txt


def compile_infos(cleaned_txt, df, url, complete_link):
    id = url.split("https://www.ijrh.org/journal/")[1]

    row = pd.DataFrame(
        {
            "category-id": "alternative_science",
            "text-id": "IJRH" + id,
            "tags": "",
            "venue": "Indian-research-Homeopathy",
            "data-source": "Indian-research-Homeopathy",
            "url": complete_link,
            "text": cleaned_txt,
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

    url_list = [
        "https://www.ijrh.org/journal/vol1/iss1/",
        "https://www.ijrh.org/journal/vol2/iss4/",
        "https://www.ijrh.org/journal/vol2/iss3/",
        "https://www.ijrh.org/journal/vol2/iss2/",
        "https://www.ijrh.org/journal/vol2/iss1/",
        "https://www.ijrh.org/journal/vol3/iss3/",
        "https://www.ijrh.org/journal/vol3/iss2/",
        "https://www.ijrh.org/journal/vol3/iss1/",
        "https://www.ijrh.org/journal/vol4/iss1/",
        "https://www.ijrh.org/journal/vol4/iss2/",
        "https://www.ijrh.org/journal/vol4/iss3/",
        "https://www.ijrh.org/journal/vol4/iss4/",
        "https://www.ijrh.org/journal/vol5/iss1/"
        "https://www.ijrh.org/journal/vol5/iss2/",
        "https://www.ijrh.org/journal/vol5/iss3/",
        "https://www.ijrh.org/journal/vol5/iss4/",
        "https://www.ijrh.org/journal/vol6/iss1/",
        "https://www.ijrh.org/journal/vol6/iss2/",
        "https://www.ijrh.org/journal/vol6/iss3/",
        "https://www.ijrh.org/journal/vol7/iss1/",
        "https://www.ijrh.org/journal/vol7/iss2/",
        "https://www.ijrh.org/journal/vol7/iss3/",
        "https://www.ijrh.org/journal/vol7/iss4/",
        "https://www.ijrh.org/journal/vol8/iss1/",
        "https://www.ijrh.org/journal/vol8/iss2/",
        "https://www.ijrh.org/journal/vol8/iss3/",
        "https://www.ijrh.org/journal/vol8/iss4/",
        "https://www.ijrh.org/journal/vol9/iss1/",
        "https://www.ijrh.org/journal/vol9/iss2/",
        "https://www.ijrh.org/journal/vol9/iss3/",
        "https://www.ijrh.org/journal/vol9/iss4/",
        "https://www.ijrh.org/journal/vol10/iss1/",
        "https://www.ijrh.org/journal/vol10/iss2/",
        "https://www.ijrh.org/journal/vol10/iss3/",
        "https://www.ijrh.org/journal/vol10/iss4/",
        "https://www.ijrh.org/journal/vol11/iss1/",
        "https://www.ijrh.org/journal/vol11/iss2/",
        "https://www.ijrh.org/journal/vol11/iss3/",
        "https://www.ijrh.org/journal/vol11/iss4/",
        "https://www.ijrh.org/journal/vol12/iss1/",
        "https://www.ijrh.org/journal/vol12/iss2/",
        "https://www.ijrh.org/journal/vol12/iss3/",
        "https://www.ijrh.org/journal/vol12/iss4/",
        "https://www.ijrh.org/journal/vol13/iss1/",
        "https://www.ijrh.org/journal/vol13/iss2/",
        "https://www.ijrh.org/journal/vol13/iss3/",
        "https://www.ijrh.org/journal/vol13/iss4/",
        "https://www.ijrh.org/journal/vol14/iss1/",
        "https://www.ijrh.org/journal/vol14/iss2/",
        "https://www.ijrh.org/journal/vol14/iss3/",
        "https://www.ijrh.org/journal/vol14/iss4/",
        "https://www.ijrh.org/journal/vol15/iss1/",
        "https://www.ijrh.org/journal/vol15/iss2/",
        "https://www.ijrh.org/journal/vol15/iss3/",
        "https://www.ijrh.org/journal/vol15/iss4/",
        "https://www.ijrh.org/journal/vol16/iss1/",
        "https://www.ijrh.org/journal/vol16/iss2/",
        "https://www.ijrh.org/journal/vol16/iss3/",
        "https://www.ijrh.org/journal/vol16/iss4/",
        "https://www.ijrh.org/journal/vol17/iss1/",
        "https://www.ijrh.org/journal/vol17/iss2/",
        "https://www.ijrh.org/journal/vol17/iss3/",
        "https://www.ijrh.org/journal/vol17/iss4/",
        "https://www.ijrh.org/journal/vol18/iss1/"
        "https://www.ijrh.org/journal/vol18/iss2/",
    ]
    df = pd.DataFrame(
        columns=[
            "category-id",
            "text-id",
            "tags",
            "venue",
            "data-source",
            "url",
            "text",
        ]
    )

    for url in url_list:
        complete_links = get_urls(url)
        print(complete_links)
        # if present download file
        for complete_link in complete_links:

            print("Downloading file: ", complete_link)
            path = download_pdf(complete_link)


            pdf_txt = pdf_to_text(path)
            if pdf_txt is None:
                continue

            cleaned_txt = clean_text(pdf_txt)

            df = compile_infos(cleaned_txt, df, url, complete_link)
    print("All PDF files downloaded")
    df.to_csv(args.output,
        index=False,
    )
    print("done")


if __name__ == "__main__":
    main()
