#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = (
    "harvest Open Citation for passive citations (cited-by). Infos on the API: https://opencitations.net/index/api/v1 "
    "how to: "
    "1) get all PMIDS related to a MESH term: harvest_PMC_citation-freq_topicwise.py"
    "2) havest citations from Open citations and calculate the ten percent most cited: harvest_PMC_citation-freq_Opencitations_for-citation.py"
    "3) get text for ten percent most cited: harvest_PMC_citation-freq_PMID-to-txt.py"
)

__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import pandas as pd
import requests
import argparse
import numpy as np
from sympy.codegen.ast import continue_


def get_doi(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        if data:
            try:
                ids = data["result"][f"{pmid}"]["articleids"]
                try:
                    doi = next(
                        (item for item in ids if item.get("idtype") == "doi"), {}
                    ).get("value")
                except Exception as e:
                    print(e)
                print("retrieved DOI:", doi)
                return doi
            except Exception as e:
                print(e)
        else:

            return None
    else:
        return None


def get_citedby(doi, token):
    API_CALL = f"https://opencitations.net/index/api/v1/references/{doi}"
    HTTP_HEADERS = {"authorization": token}

    response = requests.get(API_CALL, headers=HTTP_HEADERS)
    try:
        data = response.json()
        citedby = len(data)
        return citedby
    except Exception as e:
        print(e)
        return None


def main():
    token = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("PMID_file")
    parser.add_argument("mesh")
    parser.add_argument("output")  # containing ten most percent cited-by information
    args = parser.parse_args()

    pmid_file = open(args.PMID_file, "r")
    Lines = pmid_file.readlines()
    mesh = args.mesh

    df = pd.DataFrame(columns= ['pmid','doi','citedby'])

    for pmid in Lines:
        pmid = pmid.replace('\n', '')
        doi = get_doi(pmid)
        if doi is not None:
            citedby = get_citedby(doi, token)
            infos = {'pmid': pmid, 'doi' : doi, 'citedby': citedby}
            df = pd.concat([df, pd.DataFrame([infos])], ignore_index=True)
        else:
            print("No DOI detected.")
    df.to_csv(f'df_number_citations_{mesh}.csv')


    number_df = df.shape[0]
    ten_most = number_df / 10
    ten_most = round(ten_most)
    print("absolut number of ten percent most cited:", ten_most)

    df["citedby"] = df["citedby"].replace("None", np.nan)
    df["citedby"] = df["citedby"].replace("", "0")

    if pd.api.types.is_integer_dtype(df["citedby"]):
        pass
    elif pd.api.types.is_string_dtype(df["citedby"]):

        df["citedby"] = df["citedby"].astype(int)
    else:
        pass

    df_ten_most = df.nlargest(ten_most, "citedby")

    df_ten_most.to_csv(args.output,
        index=False,
    )

    print("done")


if __name__ == "__main__":
    main()
