import os
import csv

def generate_rfam_dictionary(workdir):
    new_workdir = workdir.parent / "ftp_input"
    rfam_file = os.path.join(new_workdir, "emdb_pdb_rfam.csv")

    rfam_dictionary = {}

    with open(rfam_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row['pdb_id']
            rfam = row['rfam']
            rfam_id = row['rfam_id']
            entity_id = row['entity_id']

            if pdb_id not in rfam_dictionary:
                rfam_dictionary[pdb_id] = {}

            rfam_dictionary[pdb_id][entity_id] = {
                'rfam': rfam,
                'rfam_id': rfam_id
            }

    return rfam_dictionary

class RfamMapping:
    """
    Map Rfam IDs to EMDB ID and sample id along with the name
    """

    def __init__(self, rfam):
        self.rfam = rfam

    def execute(self, rfam_dictionary):
        for rf in self.rfam:
            rf = self.worker(rfam_dictionary, rf)
        return self.rfam

    def worker(self, rfam_dictionary, rf):
        for model in rf.pdb_id:
            pdb_id = model.pdb_id
            if pdb_id in rfam_dictionary:
                if rf.sample_id in rfam_dictionary[pdb_id]:
                    rfam_entry = rfam_dictionary[pdb_id][rf.sample_id]
                    rfam_acc, rfam_name = rfam_entry['rfam'].split(":", 1)
                    rf.rfam_id = rfam_entry['rfam_id'].strip()
                    rf.rfam_acc = rfam_acc.strip()
                    rf.provenance = "PDBe"

    def export_tsv(self, rfam_logger):
        for rf in self.rfam:
            if hasattr(rf, 'rfam_acc') and rf.rfam_acc:
                rfam_logger.info(str(rf))