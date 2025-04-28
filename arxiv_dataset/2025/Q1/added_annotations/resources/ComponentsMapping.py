from gemmi import cif
import lxml.etree as ET
### TO DO LIST: ##
#### Replace (logger.debug(HET, "NOT IN PDB_CCD") with corresponding resource API,
##### as of now no entry has HET which is not in CCD ####

def parseCCD(cdd_file):
    """
    Extract only the external mapping for the HET_CODE from the pdbe components.cif file
    """
    chembl_map = {}
    chebi_map = {}
    drugbank_map = {}

    doc = cif.read_file(cdd_file)
    for block in doc:
        HET = block.name
        name = block.find_value('_chem_comp.name')
        for db,value in block.find('_pdbe_chem_comp_external_mappings.', ['resource', 'resource_id']):
            if db == "ChEMBL":
                chembl_map[HET] = value
            elif db == "ChEBI":
                chebi_map[HET] = value
            elif db == "DrugBank":
                drugbank_map[HET] = value
    return chembl_map, chebi_map, drugbank_map

class ComponentsMapping:
    """
    Extracting PDB_IDs from header and get the HET code from PDBe components.cif file.
    Querying with the extracted HET code for mapping the EMDB entries to the Chemical Component Dictionary (CCD) for
    mapping to various database like ChEMBL, ChEBI and DrugBank.
    """

    def __init__(self, ligands):
        self.ligands = ligands

    def execute(self, chembl_map, chebi_map, drugbank_map):
        ###### Mapping HET_CODE TO CHEMBL, CHEBI and DRUGBANK ########
        for ligand in self.ligands:
            ligand = self.worker(ligand, chembl_map, chebi_map, drugbank_map)
        return self.ligands

    def worker(self, ligand, chembl_map, chebi_map, drugbank_map):
        HET = ligand.HET
        if HET in chembl_map:
            ligand.chembl_id = chembl_map[HET]
            ligand.provenance_chembl = "PDBe-CCD"
        if HET in chebi_map:
            ligand.chebi_id = chebi_map[HET]
            ligand.provenance_chebi = "PDBe-CCD"
        if HET in drugbank_map:
            ligand.drugbank_id = drugbank_map[HET]
            ligand.provenance_drugbank = "PDBe-CCD"
        return ligand

    def export_tsv(self, chembl_logger, chebi_logger, drugbank_logger):
        for ligand in self.ligands:
            if ligand.provenance_chembl:
                chembl_logger.info(ligand.get_chembl_tsv())
            if ligand.provenance_chebi:
                chebi_logger.info(ligand.get_chebi_tsv())
            if ligand.provenance_drugbank:
                drugbank_logger.info(ligand.get_drugbank_tsv())
