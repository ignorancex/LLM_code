import unittest
import io
import resources.PublicationMapping
from models import Citation, Author
import mock

class TestPublicationMapping(unittest.TestCase):
    """
    UnitTest for Citation annotations
    """
    def setUp(self):
        super(TestPublicationMapping, self).setUp()
        self.citation = Citation("EMD-13501", pmedid="34812732", authors=[Author('Wang A', 1), Author('Tani B', 2)],
                                     title="High-resolution structures of the actomyosin-V complex .", provenance_pm="EMDB")
        self.pubmed_dict = {'34812732': {'pmid': '34812732', 'pmcid': 'PMC8735999', 'doi': '10.7554/elife.73724', 'issn': '2050-084X',
                                         'journal': 'Elife', 'journal_abbv': 'Elife', 'authors': ['0000-0002-5119-3039', '0000-0002-6290-8853']},
                            '21224846': {'pmid': '21224846', 'pmcid': 'PMC3105306', 'doi': '10.1038/ncomms1154', 'issn': '2041-1723',
                                         'journal': 'Nature Communication', 'journal_abbv': 'NAT.COMMUN.', 'authors': ['0000-0003-2681-5921', '0000-0003-4835-154X', '0000-0002-8070-1493']}
                            }
        self.pmedid = '34812732'
        self.pmcid = 'PMC8735999'
        self.doi = '10.7554/elife.73724'
        self.issn = '2050-084X'
        self.authors = [('0000-0002-5119-3039', 1, 'EuropePMC'), ('0000-0002-6290-8853', 2, 'EuropePMC')]
        self.provenance = "EuropePMC"
        self.journal = 'Elife'
        self.journal_abbv = 'Elife'

    @mock.patch("builtins.open")
    def test_generate_pubmed_dictionary(self, open_mock):
        open_mock.return_value = io.StringIO(
        "PMID\tPMC\tDOI\tISSN\t[AUTHORS ORCID]\n"
        "34812732\tPMC8735999\t10.7554/elife.73724\t2050-084X\tElife\tElife\t0000-0002-5119-3039	0000-0002-6290-8853\n"
        "21224846\tPMC3105306\t10.1038/ncomms1154\t2041-1723\tNature Communication\tNAT.COMMUN.\t0000-0003-2681-5921	0000-0003-4835-154X	0000-0002-8070-1493\n")
        self.assertEqual(resources.PublicationMapping.generate_pubmed_dictionary(open_mock), self.pubmed_dict)

    def test_worker(self):
        CitationMap = resources.PublicationMapping.PublicationMapping(self.citation)
        pub = CitationMap.worker(self.pubmed_dict)
        self.assertEqual(pub.pmedid, self.pmedid)
        self.assertEqual(pub.pmcid, self.pmcid)
        self.assertEqual(pub.provenance_pmc, self.provenance)
        self.assertEqual(pub.doi, self.doi)
        self.assertEqual(pub.provenance_doi, self.provenance)
        self.assertEqual(pub.issn, self.issn)
        self.assertEqual(pub.provenance_issn, self.provenance)
        self.assertEqual(pub.journal, self.journal)
        self.assertEqual(pub.journal_abbv, self.journal_abbv)
        for n in range(len(pub.authors)):
            self.assertEqual(pub.authors[n].orcid, self.authors[n][0])
            self.assertEqual(pub.authors[n].order, self.authors[n][1])
            self.assertEqual(pub.authors[n].provenance, self.authors[n][2])