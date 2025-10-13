#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "compile harvested PMC files how to:  1) get all PMIDS related to a MESH term: harvest_PMC_citation-freq_topicwise.py, 2) havest citations from Open citations and calculate the ten percent most cited: harvest_PMC_citation-freq_Opencitations_for-citation.py 3) get text for ten percent most cited: harvest_PMC_citation-freq_PMID-to-txt.py, 4) 4. compile files to one and delete empty texts: compile_PMC_citation-freq_data.py"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2ß25 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import glob
import pandas as pd
import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_folder")
    argparser.add_argument("output")
    args = argparser.parse_args()


    df_list = []
    path = args.input_folder
    pdfs = glob.glob(f'{path}/*')

    for pdf in pdfs:
        pdf_ = pd.read_csv(pdf)
        df_list.append(pdf_)


    df = pd.concat(df_list, ignore_index=True)

    df = df.dropna(subset=["text"])
    df.drop(df[df["text"].str.len() < 20].index, inplace=True)

    df.to_csv(args.output)


if __name__ == "__main__":
    main()
