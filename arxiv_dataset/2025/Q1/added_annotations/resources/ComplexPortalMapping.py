import os, csv
from glob import glob
from models import CPX, EMDB_complex
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(funcName)s:%(message)s')
file_handler = logging.FileHandler('logging_ComplexPortal.log', mode ='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

MIN_SCORE = 0.5

def overlap(set1, set2):
    L = len(set1.union(set2))
    match = len(set1.intersection(set2))
    return float(match)/float(L)


class CPX_database:
    """
    Class containing all the CPX entries from their FTP area
    """
    def __init__(self):
        self.entries = {}
        self.id_mapped = {}
        self.uniprot_map = {}

    def add_row(self, row):
        cpx = CPX(row)
        self.entries[cpx.cpx_id] = cpx
        for identifier in cpx.identifiers:
            if identifier in self.id_mapped:
                self.id_mapped[identifier].add(cpx.cpx_id)
            else:
                self.id_mapped[identifier] = set([cpx.cpx_id])
        for unp_id in cpx.uniprot:
            if unp_id in self.uniprot_map:
                self.uniprot_map[unp_id].add(cpx.cpx_id)
            else:
                self.uniprot_map[unp_id] = set([cpx.cpx_id])

    def get_from_cpx(self, cpx_id):
        if cpx_id in self.entries:
            return self.entries[cpx_id]
        return None

    # Use this method to return CPX entry based on Uniprot, CHEBI or PubMed ids
    def get_from_identifier(self, ext_id):
        if ext_id in self.id_mapped:
            return self.id_mapped[ext_id]
        return None

    # Use this method to return CPX entry based on Uniprot ID
    def get_from_uniprot(self, unp_id):
        if unp_id in self.uniprot_map:
            return self.uniprot_map[unp_id]
        return None

class CPMapping:
    """
    Extracting the IDs from the EMDB entry and for map only entries running the BLASTP search to get UNIPROT IDs.
    Querying with the extracted IDs for mapping the EMDB entries to the complex portal if exists.
    """

    def __init__(self, proteins, supras, CP_ftp):
        self.cpx_db = CPX_database()
        self.emdb_complexes = {}
        self.annotations = []
        self.CP_ftp = CP_ftp

        # Parse Complex Portal tables
        for fn in glob(os.path.join(str(self.CP_ftp), '*.tsv')):
            with open(fn, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)  # skip the headers
                batch_data = list(reader)
                for line in batch_data:
                    self.cpx_db.add_row(line)

        # Parse EMDB proteins
        for protein in proteins:
            #protein is part of a complex and contains a Uniprot ID
            if len(protein.sample_complexes) > 0 and protein.uniprot_id:
                emdb_id = protein.emdb_id
                sample_id = protein.sample_id
                uniprot_id = protein.uniprot_id
                supra_name = []
                for supra in supras:
                    if supra.emdb_id == emdb_id:
                        supra_name.append(supra.name)
                sample_copies = protein.sample_copies
                for complex_id in protein.sample_complexes:
                    emdb_complex_id = emdb_id + "_" + str(complex_id)
                    for name in supra_name:
                        if emdb_complex_id in self.emdb_complexes:
                            self.emdb_complexes[emdb_complex_id].add_protein(uniprot_id)
                        else:
                            emdb_cpx = EMDB_complex(emdb_id, complex_id, name, sample_copies, emdb_complex_id)
                            emdb_cpx.add_protein(uniprot_id)
                            self.emdb_complexes[emdb_complex_id] = emdb_cpx

    def execute(self):
        for emdb_complex in self.emdb_complexes.values():
            self.annotations.append(self.worker(emdb_complex))
        return self.annotations

    def worker(self, emdb_complex):
        emdb_protein_list = emdb_complex.proteins
        cpx_found = set()
        for uniprot_id in emdb_protein_list:
            cpx_complexes = self.cpx_db.get_from_uniprot(uniprot_id)
            if cpx_complexes:
                for cpx_id in cpx_complexes:
                    cpx_found.add(cpx_id)

        max_score = 0.0
        best_hits = []
        for cpx_id in cpx_found:
            cpx = self.cpx_db.get_from_cpx(cpx_id)
            cpx_uniprot = cpx.uniprot
            overlap_score = overlap(emdb_protein_list, cpx_uniprot)
            if overlap_score == max_score:
                best_hits.append(cpx)
            elif overlap_score > max_score:
                best_hits = [cpx]
                max_score = overlap_score

        if max_score >= MIN_SCORE:
            emdb_complex.cpx_list = best_hits
            emdb_complex.score = max_score
            emdb_complex.provenance = "Complex Portal"

            return emdb_complex
        return None

    def export_tsv(self, cpx_logger):
        for emcpx in self.annotations:
            if emcpx:
                for cpx in emcpx.cpx_list:
                    cpx_logger.info("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (emcpx.emdb_id, (emcpx.sample_id).split("_")[1], emcpx.name.replace("\n", ""),
                                                                  emcpx.sample_copies, cpx.cpx_id, cpx.name, emcpx.provenance, emcpx.score))