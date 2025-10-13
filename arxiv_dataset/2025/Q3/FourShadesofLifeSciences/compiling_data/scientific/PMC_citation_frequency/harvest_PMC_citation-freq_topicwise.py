#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "harvest PMID via Entrez-API from MEDLINE data directory"
                    "how to: "
                   "1) get all PMIDS related to a MESH term: harvest_PMC_citation-freq_topicwise.py"
                   "2) havest citations from Open citations and calculate the ten percent most cited: harvest_PMC_citation-freq_Opencitations_for-citation.py"
                   "3) get text for ten percent most cited: harvest_PMC_citation-freq_PMID-to-txt.py"
                    "4) 4. compile files to one and delete empty texts: compile_PMC_citation-freq_data.py"
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
    runs = int(count)/1000
    runs += 1

    for i in range(int(runs)):
        item_numer_start = 1000*i+1
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
