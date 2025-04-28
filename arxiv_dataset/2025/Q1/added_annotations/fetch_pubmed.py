import argparse, configparser, os, json
from pathlib import Path
import requests
from glob import glob
import lxml.etree as ET

class PubRef:
    def __init__(self, pubmed, pmc="", doi="", issn="", journal="", abbv=""):
        self.pubmed = pubmed
        self.doi = doi
        self.issn = issn
        self.pmc = pmc
        self.journal = journal
        self.journal_abbreviation = abbv
        self.authors = []
    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'pubmed', None) == self.pubmed and
            getattr(other, 'doi', None) == self.doi and
            getattr(other, 'pmc', None) == self.pmc and
            getattr(other, 'issn', None) == self.issn)
    def __hash__(self):
        return hash(self.pubmed + self.doi + self.issn + self.pmc) + hash(self.authors)
    def __str__(self):
        if len(self.authors) == 0:
            return f"{self.pubmed}\t{self.pmc}\t{self.doi}\t{self.issn}\t{self.journal}\t{self.journal_abbreviation}"
        return f"{self.pubmed}\t{self.pmc}\t{self.doi}\t{self.issn}\t{self.journal}\t{self.journal_abbreviation}\t" + '\t'.join(self.authors)

def call_ePubmedCentral(pubmed_list, uri):
    publications = []

    for i in range(0, len(pubmed_list), N):
        subset = pubmed_list[i:i+N]
        query = ' OR '.join(subset)
        queryString = f"SRC:MED AND EXT_ID:({query})"
        data = {'query': queryString,
                'format': 'json',
                'resultType': 'core',
                "pageSize": N}
        response = requests.post(uri, data=data)
        if response.status_code == 200:
            try:
                pmcjdata = json.loads(response.text)
                if 'result' in pmcjdata['resultList']:
                    result = pmcjdata['resultList']['result']
                    for pub_data in result:
                        pmid = pub_data['id']
                        pmcid = pub_data['pmcid'] if "pmcid" in pub_data else ""
                        doi = pub_data['doi'] if "doi" in pub_data else ""
                        issn = ""
                        journal_name = ""
                        abbv = ""
                        if "journalInfo" in pub_data:
                            if "journal" in pub_data['journalInfo']:
                                issn = pub_data['journalInfo']['journal']['issn'] if 'issn' in pub_data['journalInfo']['journal'] else ""
                                journal_name = pub_data['journalInfo']['journal']['title'] if 'title' in pub_data['journalInfo']['journal'] else ""
                                abbv = pub_data['journalInfo']['journal']['medlineAbbreviation'] if 'medlineAbbreviation' in pub_data['journalInfo']['journal'] else ""
                        pub_ref = PubRef(pmid, pmcid, doi, issn, journal_name, abbv)
                        if "authorList" in pub_data:
                            for author in pub_data['authorList']['author']:
                                if "authorId" in author:
                                    if author['authorId']['type'] == "ORCID":
                                        pub_ref.authors.append(author['authorId']['value']) #Found ORCID
                                    else:
                                        pub_ref.authors.append("") #No ORCID
                                else:
                                    pub_ref.authors.append("") #No ORCID
                        publications.append(pub_ref)

            except json.JSONDecodeError:
                print(f"WARN: Failed to read the result of {queryString}")
        else:
            print(f"WARN: Failed to connect to Europe PMC on {queryString}")
    return publications


def get_pubmed_ids(header_dir):
    pubmed_list = set()
    for xml_dirpath in glob(os.path.join(str(header_dir), '*')):
        split_dir = xml_dirpath.split('-')
        if len(split_dir) == 2:
            id_num = xml_dirpath.split('-')[1]
            emdb_id = f"EMD-{id_num}"
            xml_filepath = os.path.join(xml_dirpath, f"header/emd-{id_num}-v30.xml")
            if not os.path.isfile(xml_filepath):
                print(f"{xml_filepath} not found.")
                continue
            tree = ET.parse(xml_filepath)

            xrefs = tree.xpath("//crossreferences/citation_list/primary_citation/*/external_references")
            for ref in xrefs:
                if ref.attrib['type'] == "PUBMED":
                    pubmed_list.add(ref.text)
    
    return list(pubmed_list)

if __name__ == "__main__":
    prog = "Publication (EMICSS)"
    usage = """
            Collect Publication and author EMICSS
            Example:
            python fetch_pubmed.py -w '[{"/path/to/working/folder"}]'
            -f '[{"/path/to/EMDB/header/files/folder"}]' -N 400
          """

    parser = argparse.ArgumentParser(prog=prog, usage=usage, add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument('-w', '--workDir', type=Path, help="Main working directory path .")
    parser.add_argument('-f', '--headerDir', type=Path, help="Directory path to the EMDB version 3.0 header files.")
    parser.add_argument('-N', default=500, help="Number of simultaneosly papers to be included in a query.")

    args = parser.parse_args()
    workDir = args.workDir
    headerDir = args.headerDir
    N = int(args.N)

    #Get config variables:
    config = configparser.ConfigParser()
    env_file = os.path.join(Path(__file__).parent.absolute(), "config.ini")
    config.read(env_file)
    pmc_uri = config.get("api", "pmc")

    pubmed_list = get_pubmed_ids(headerDir)
    publications = call_ePubmedCentral(pubmed_list, pmc_uri)

    #Export results
    with open(os.path.join(workDir, "EPMC_pubmed.tsv"), "w") as fw:
        fw.write("PMID\tPMC\tDOI\tISSN\tJOURNAL\tABBREVIATION\t[AUTHORS ORCID]\n")
        for pub in publications:
            fw.write(f"{str(pub)}\n")
