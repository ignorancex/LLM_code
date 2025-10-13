#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = ("harvest PMID by MESH via Entrez-API from MEDLINE data directory"
                   "MESH term:"
                    "Dementia [C10.228.140.380, F03.615.400]"
                    "Myocardial Infarction [C14.280.647.500, C14.907.585.500, C23.550.513.355.750, C23.550.717.489.750]"
                    "Sleep Initiation and Maintenance Disorders [C10.886.425.800.800, F03.870.400.800.800]"
                    "Menopause [G08.686.157.500, G08.686.841.249.500]"
                    "Stroke [C10.228.140.300.775, C14.907.253.855]"
                    "Tobacco Use [F01.145.958] / Tobacco Smoking [F01.145.805.375,  F01.145.958.875]"
                    "Curcuma [B01.875.800.575.912.250.618.937.900.166]"
                    "Measles [C01.925.782.580.600.500.500]"
                    "Inflammation [C23.550.470]"
                    "Vaccines [D20.215.894]"
                    "Abortion, Induced [E04.520.050]
                    "Climate change  [G16.500.175.374]"
                    "Pandemics [N06.850.290.200.600]"
                    "Urine [A12.207.927]")
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import requests
import argparse


def get_PMID(file, mesh):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={mesh}[MeSH%20Terms]+AND+medline[sb]&retmode=json&retmax=1000"
    response = requests.get(url)
    data = response.json()

    # Extract PMIDs
    count = data["esearchresult"]['count']
    runs = int(count) / 1000
    runs += 1

    for i in range(int(runs)):
        item_numer_start = 1000 * i + 1
        print(item_numer_start)

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={mesh}[MeSH%20Terms]+AND+medline[sb]&retmode=json&retmax=1000&retstart={item_numer_start}"
        response = requests.get(url)
        data = response.json()

        # Extract PMIDs
        try:
            pmids = data["esearchresult"]["idlist"]
        except Exception as e:
            print(e)

        # Save PMIDs to a text file
        try:
            for pmid in pmids:
                file.write(pmid + "\n")
        except Exception as e:
            print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_term')
    args = parser.parse_args()
    mesh = args.mesh_term
    mesh_stripped = mesh.strip()
    print(mesh_stripped)

    with open(f"pmids_{mesh_stripped}.txt", "w") as file:
        get_PMID(file, mesh)
    print('done')


if __name__ == '__main__':
    main()
