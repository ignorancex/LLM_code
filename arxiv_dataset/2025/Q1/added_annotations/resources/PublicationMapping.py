import os
import itertools

def generate_pubmed_dictionary(workDir):
    pubmed_dict = {}
    epmc_pubmed = os.path.join(workDir, "EPMC_pubmed.tsv")
    with open(epmc_pubmed, 'r') as f:
        for line in f.readlines()[1:]:
            row = line.strip('\n').split('\t')
            pubmed_dict[row[0]] = {'pmid': row[0], 'pmcid': row[1], 'doi': row[2], 'issn': row[3], 'journal': row[4], 'journal_abbv': row[5], 'authors': row[6:]}
    return pubmed_dict

class PublicationMapping:
    """
     If pubmed id available then annotations collected from EuropePMC API (PubMed, PubMED central, DOI and ORCID IDs),
     if no pubmed id then author provided annotations for the publication.
    """

    def __init__(self, citation):
        self.citation = citation

    def execute(self, pubmed_dict):
        self.worker(pubmed_dict)
        return self.citation

    def worker(self, pubmed_dict):
        if self.citation.pmedid:
            pmid = self.citation.pmedid
            if pmid in pubmed_dict:
                pm = pubmed_dict[pmid]
                
                # ORCID
                authors = pm['authors']
                for i, orcid in enumerate(authors):
                    if orcid:
                        self.citation.addExternalOrcid(orcid, i+1, "EuropePMC")

                # PMC
                if pm['pmcid']:
                    self.citation.pmcid = pm['pmcid']
                    self.citation.provenance_pmc = "EuropePMC"
                if pm['doi']:
                    self.citation.doi = pm['doi']
                    self.citation.provenance_doi = "EuropePMC"
                if pm['issn']:
                    self.citation.issn = pm['issn']
                    self.citation.provenance_issn = "EuropePMC"

                if pm['journal']:
                    self.citation.journal = pm['journal']
                if pm['journal_abbv']:
                    self.citation.journal_abbv = pm['journal_abbv']
        return self.citation


    def export_tsv(self, pubmed_logger, orcid_logger, author_logger):
        if self.citation.pmedid or self.citation.doi:
            pubmed_logger.info(str(self.citation))
        for author in self.citation.authors:
            author_logger.info(f"{self.citation.emdb_id}\t{str(author)}")
            if author.orcid:
                orcid_logger.info(f"{self.citation.emdb_id}\t{str(author)}")
