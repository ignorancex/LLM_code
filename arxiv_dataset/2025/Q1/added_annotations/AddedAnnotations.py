import argparse, configparser, os, json
from pathlib import Path
import models
from resources.ComplexPortalMapping import CPMapping
from resources.ComponentsMapping import ComponentsMapping, parseCCD
from resources.UniprotMapping import UniprotMapping, generate_unp_dictionary
from resources.StructureMapping import StructureMapping
from resources.PublicationMapping import PublicationMapping, generate_pubmed_dictionary
from resources.ProteinTermsMapping import ProteinTermsMapping
from resources.RfamMapping import RfamMapping, generate_rfam_dictionary
from XMLParser import XMLParser
from glob import glob
import logging
from joblib import Parallel, delayed
formatter = logging.Formatter('%(message)s')


def setup_logger(name, log_file, level=logging.INFO, mode='w'):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode=mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def start_logger_if_necessary(log_name, log_file):
    logger = logging.getLogger(log_name)
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, mode='a')
        logger.addHandler(fh)
    return logger

def read_json(input_file, headerDir):
    entries = set()
    files = []
    with open(input_file) as json_file:
        data = json.load(json_file)
        if 'mapReleases' in data:
            if 'entries' in data['mapReleases']:
                entries.update(data['mapReleases']['entries'])
    for entry in entries:
        files.append(os.path.join(headerDir, entry))

    return files

def run(filename):
    id_num = filename.split('-')[1]
    print(f"Running EMD-{id_num}")
    xml_filepath = os.path.join(filename, f"header/emd-{id_num}-v30.xml")
    if not os.path.isfile(xml_filepath):
        print(f"{xml_filepath} not found.")
        return None
    xml = XMLParser(xml_filepath)
    packed_models['HEADER'] = xml
    if uniprot:
        uniprot_log = start_logger_if_necessary("uniprot_logger", uniprot_log_file)
        unp_mapping = UniprotMapping(args.workDir, xml.proteins, uniprot_dictionary, blast_db, blastp_bin)
        unip_map = unp_mapping.execute()
        unp_mapping.export_tsv(uniprot_log)
        packed_models['UNIPROT'] = unip_map
    if cpx:
        cpx_logger = start_logger_if_necessary("cpx_logger", cpx_log_file)
        cpx_mapping = CPMapping(unp_mapping.proteins, xml.supramolecules, CP_ftp)
        cpx_map = cpx_mapping.execute()
        cpx_mapping.export_tsv(cpx_logger)
        packed_models["COMPLEX"] = cpx_map
    if component:
        chembl_log = start_logger_if_necessary("chembl_logger", chembl_log_file)
        chebi_log = start_logger_if_necessary("chebi_logger", chebi_log_file)
        drugbank_log = start_logger_if_necessary("drugbank_logger", drugbank_log_file)
        comp_mapping = ComponentsMapping(xml.ligands)
        comp_map = comp_mapping.execute(chembl_map, chebi_map, drugbank_map)
        comp_mapping.export_tsv(chembl_log, chebi_log, drugbank_log)
        packed_models["LIGANDS"] = comp_map
    if model:
        model_logger = start_logger_if_necessary("model_logger", model_log_file)
        mw_mapping = StructureMapping(xml.models, assembly_ftp)
        mw_map = mw_mapping.execute()
        mw_mapping.export_tsv(model_logger)
        packed_models["MODEL"] = mw_map
    if weight:
        weight_logger = start_logger_if_necessary("weight_logger", weight_log_file)
        weight_logger.info(f"{xml.emdb_id}\t{xml.overall_mw}")
        wgt = models.Weight(xml.emdb_id)
        (wgt.emdb_id, wgt.overall_mw, wgt.units, wgt.provenance) = (xml.emdb_id, xml.overall_mw, "MDa", "EMDB")
        packed_models["WEIGHT"] = wgt
    if pmc or orcid:
        pubmed_log = start_logger_if_necessary("pubmed_logger", pubmed_log_file) if pmc else None
        orcid_log = start_logger_if_necessary("orcid_logger", orcid_log_file) if orcid else None
        author_log = start_logger_if_necessary("author_logger", author_log_file) if pmc else None
        pmc_mapping = PublicationMapping(xml.citation)
        pmc_map = pmc_mapping.execute(pubmed_dict)
        pmc_mapping.export_tsv(pubmed_log, orcid_log, author_log)
        packed_models["CITATION"] = pmc_map
    if go or interpro or pfam or cath or scop or scop2 or scop2B or pdbekb:
        go_log = start_logger_if_necessary("go_logger", go_log_file) if go else None
        interpro_log = start_logger_if_necessary("interpro_logger", interpro_log_file)  if interpro else None
        pfam_log = start_logger_if_necessary("pfam_logger", pfam_log_file)  if pfam else None
        cath_log = start_logger_if_necessary("cath_logger", cath_log_file)  if cath else None
        scop_log = start_logger_if_necessary("scop_logger", scop_log_file) if scop else None
        scop2_log = start_logger_if_necessary("scop2_logger", scop2_log_file) if scop2 else None
        scop2B_log = start_logger_if_necessary("scop2B_logger", scop2B_log_file) if scop2B else None
        pdbekb_log = start_logger_if_necessary("pdbekb_logger", pdbekb_log_file) if pdbekb else None
        PT_mapping = ProteinTermsMapping(unp_mapping.proteins, sifts_path, go, interpro, pfam, cath, scop, scop2, scop2B, pdbekb)
        proteins_map = PT_mapping.execute(uniprot_with_models)
        PT_mapping.export_tsv(go_log, interpro_log, pfam_log, cath_log, scop_log, scop2_log, scop2B_log, pdbekb_log)
        packed_models["PROTEIN-TERMS"] = proteins_map
    if rfam:
        rfam_logger = start_logger_if_necessary("rfam_logger", rfam_log_file)
        rfam_mapping = RfamMapping(xml.rfams)
        rfam_map = rfam_mapping.execute(rfam_dictionary)
        rfam_mapping.export_tsv(rfam_logger)
        packed_models["RFAM"] = rfam_map

"""
List of things to do:
  - Adapt the unit tests to work with this version
"""

if __name__ == "__main__":
    ######### Command : python /Users/amudha/project/git_code/added_annotations/AddedAnnotations.py
    # -w /Users/amudha/project/ -f /Users/amudha/project/EMD_XML/ --CPX --model --uniprot --weight --pmc

    prog = "EMDBAddedAnnotations"
    usage = """
            Mapping EMDB entries to Complex portal, UNIPROT, chEMBL.
            Example:
            python AddedAnnotations.py -w '[{"/path/to/working/folder"}]'
            -f '[{"/path/to/EMDB/header/files/folder"}]'
            --uniprot --CPX --component --model --weight --pmc --GO --interpro --pfam --pbdekb 
            --cath --scop --scop2 --scop2B --rfam
          """

    parser = argparse.ArgumentParser(prog=prog, usage=usage, add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument('-w', '--workDir', type=Path, help="Main working directory path .")
    parser.add_argument('-f', '--headerDir', type=Path, help="Directory path to the EMDB version 3.0 header files.")
    parser.add_argument('-t', '--threads', type=int, default=4, help="Number of threads.")
    parser.add_argument('--json', type=Path, help="Path to release json file.")
    parser.add_argument("--all", type=bool, nargs='?', const=True, default=False, help="Fetch all external resources.")
    parser.add_argument("--uniprot", type=bool, nargs='?', const=True, default=False, help="Mapping to Complex Portal.")
    parser.add_argument("--CPX", type=bool, nargs='?', const=True, default=False, help="Mapping to Complex Portal.")
    parser.add_argument("--component", type=bool, nargs='?', const=True, default=False, help="Mapping to ChEMBL, ChEBI and DrugBank.")
    parser.add_argument("--model", type=bool, nargs='?', const=True, default=False, help="Collect MW from PDBe.")
    parser.add_argument("--weight", type=bool, nargs='?', const=True, default=False, help="Collect sample weight from header file.")
    parser.add_argument("--pmc", type=bool, nargs='?', const=True, default=False, help="Mapping publication ID to EMDB entries")
    parser.add_argument("--GO", type=bool, nargs='?', const=True, default=False, help="Mapping GO ids to EMDB entries")
    parser.add_argument("--interpro", type=bool, nargs='?', const=True, default=False, help="Mapping InterPro ids to EMDB entries")
    parser.add_argument("--pfam", type=bool, nargs='?', const=True, default=False, help="Mapping pfam ids to EMDB entries")
    parser.add_argument("--cath", type=bool, nargs='?', const=True, default=False, help="Mapping Cath domains to EMDB entries")
    parser.add_argument("--scop", type=bool, nargs='?', const=True, default=False, help="Mapping SCOP domains to EMDB entries")
    parser.add_argument("--scop2", type=bool, nargs='?', const=True, default=False, help="Mapping SCOP2 domains to EMDB entries")
    parser.add_argument("--scop2B", type=bool, nargs='?', const=True, default=False, help="Mapping SCOP2B domains to EMDB entries")
    parser.add_argument("--pdbekb", type=bool, nargs='?', const=True, default=False, help="Mapping PDBeKB links to EMDB entries")
    parser.add_argument("--rfam", type=bool, nargs='?', const=True, default=False, help="Mapping Rfam to EMDB entries")
    args = parser.parse_args()

    packed_models = {}
    db_list= []

    uniprot = args.uniprot
    cpx = args.CPX
    component = args.component
    model = args.model
    weight = args.weight
    pmc = args.pmc
    orcid = pmc
    go = args.GO
    interpro = args.interpro
    pfam = args.pfam
    cath = args.cath
    scop = args.scop
    scop2 = args.scop2
    scop2B = args.scop2B
    pdbekb = args.pdbekb
    rfam = args.rfam
    input_json = args.json
    uniprot_dictionary = {}

    if model:
        db_list.append("pdbe")
    if uniprot:
        db_list.append("uniprot")
    if component:
        db_list.append("chembl, chebi, drugbank")
    if pmc:
        db_list.append("pubmed, pubmedcentral, issn, ORCID")
    if cpx:
        uniprot = True
        db_list.append("cpx")
    if go:
        uniprot = True
        db_list.append("go")
    if interpro:
        uniprot = True
        db_list.append("interpro")
    if pfam:
        uniprot = True
        db_list.append("pfam")
    if cath:
        uniprot = True
        db_list.append("cath")
    if scop:
        uniprot = True
        db_list.append("scop")
    if scop2:
        uniprot = True
        db_list.append("scop2")
    if scop2B:
        uniprot = True
        db_list.append("scop2B")
    if pdbekb:
        uniprot = True
        db_list.append("pdbekb")
    if rfam:
        db_list.append("rfam")
    if args.all:
        uniprot = True
        cpx = True
        component = True
        model = True
        weight = True
        pmc = True
        orcid = True
        go = True
        interpro = True
        pfam = True
        cath = True
        scop = True
        scop2 = True
        scop2B = True
        pdbekb = True
        rfam = True
        db_list.extend(["pdbe", "uniprot", "chembl", "chebi", "drugbank", "pubmed", "pubmedcentral", "issn",
                        "orcid", "cpx", "go", "interpro", "pfam", "cath", "scop", "scop2", "scop2B", "pdbekb", "rfam"])

    #Get config variables:
    config = configparser.ConfigParser()
    env_file = os.path.join(Path(__file__).parent.absolute(), "config.ini")
    config.read(env_file)
    blast_db = config.get("file_paths", "BLAST_DB")
    blastp_bin = config.get("file_paths", "BLASTP_BIN")
    CP_ftp = config.get("file_paths", "CP_ftp")
    components_cif = config.get("file_paths", "components_cif")
    assembly_ftp = config.get("file_paths", "assembly_ftp")
    #emdb_empiar_list = config.get("file_paths", "emdb_empiar_list")
    pmc_api = config.get("api", "pmc")
    uniprot_tab = config.get("file_paths", "uniprot_tab")
    #GO_obo = config.get("file_paths", "GO_obo")
    sifts_path = config.get("file_paths", "sifts")

    #Start loggers
    if uniprot:
        uniprot_log_file = os.path.join(args.workDir, 'emdb_uniprot.log')
        uniprot_log = setup_logger('uniprot_logger', uniprot_log_file)
        uniprot_log.info("EMDB_ID\tSAMPLE_ID\tSAMPLE_NAME\tSAMPLE_COPIES\tNCBI_ID\tUNIPROT_ID\tPROVENANCE\tSAMPLE_COMPLEX_IDS")
    if cpx:
        cpx_log_file = os.path.join(args.workDir, 'emdb_cpx.log')
        cpx_log = setup_logger('cpx_logger', cpx_log_file)
        cpx_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tSAMPLE_NAME\tSAMPLE_COPIES\tCPX_ID\tCPX_TITLE\tPROVENANCE\tSCORE")
    if component:
        chembl_log_file = os.path.join(args.workDir, 'emdb_chembl.log')
        chebi_log_file = os.path.join(args.workDir, 'emdb_chebi.log')
        drugbank_log_file = os.path.join(args.workDir, 'emdb_drugbank.log')
        chembl_log = setup_logger('chembl_logger', chembl_log_file)
        chembl_log.info("EMDB_ID\tSAMPLE_ID\tHET_CODE\tCOMP_NAME\tCOMP_COPIES\tChEMBL_ID\tPROVENANCE")
        chebi_log = setup_logger('chebi_logger', chebi_log_file)
        chebi_log.info("EMDB_ID\tSAMPLE_ID\tHET_CODE\tCOMP_NAME\tCOMP_COPIES\tChEBI_ID\tPROVENANCE")
        drugbank_log = setup_logger('drugbank_logger', drugbank_log_file)
        drugbank_log.info("EMDB_ID\tSAMPLE_ID\tHET_CODE\tCOMP_NAME\tCOMP_COPIES\tDRUGBANK_ID\tPROVENANCE")
    if model:
        model_log_file = os.path.join(args.workDir, 'emdb_model.log')
        model_log = setup_logger('model_logger', model_log_file)
        model_log.info("EMDB_ID\tPDB_ID\tASSEMBLY\tMOLECULAR_WEIGHT")
    if weight:
        weight_log_file = os.path.join(args.workDir, 'overall_mw.log')
        weight_log = setup_logger('weight_logger', weight_log_file)
        weight_log.info("EMDB_ID\tOVERALL_MW")
    if pmc:
        pubmed_log_file = os.path.join(args.workDir, 'emdb_pubmed.log')
        pubmed_log = setup_logger('pubmed_logger', pubmed_log_file)
        pubmed_log.info("EMDB_ID\tPUBMED_ID\tPUBMED_PROVENANCE\tPUBMEDCENTRAL_ID\tPUBMEDCENTRAL_PROVENANCE\tISSN\tISSN_PROVENANCE\tDOI\tDOI_PROVENANCE\tJOURNAL_NAME\tJOURNAL_ABBV")
        author_log_file = os.path.join(args.workDir, 'emdb_author.log')
        author_log = setup_logger('author_logger', author_log_file)
        author_log.info("EMDB_ID\tAUTHOR_NAME\tAUTHOR_ORDER\tPROVENANCE")
    if orcid:
        orcid_log_file = os.path.join(args.workDir, 'emdb_orcid.log')
        orcid_log = setup_logger('orcid_logger', orcid_log_file)
        orcid_log.info("EMDB_ID\tAUTHOR_NAME\tORCID_ID\tAUTHOR_ORDER\tPROVENANCE")
    if go:
        go_log_file = os.path.join(args.workDir, 'emdb_go.log')
        go_log = setup_logger('go_logger', go_log_file)
        go_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tGO_ID\tGO_NAMESPACE\tGO_TYPE\tPROVENANCE")
    if interpro:
        interpro_log_file = os.path.join(args.workDir, 'emdb_interpro.log')
        interpro_log = setup_logger('interpro_logger', interpro_log_file)
        interpro_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tINTERPRO_ID\tINTERPRO_NAMESPACE\tSTART\tEND\tUNIPROT_START\tUNIPROT_END\tPROVENANCE")
    if pfam:
        pfam_log_file = os.path.join(args.workDir, 'emdb_pfam.log')
        pfam_log = setup_logger('pfam_logger', pfam_log_file)
        pfam_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tPFAM_ID\tPFAM_NAMESPACE\tSTART\tEND\tUNIPROT_START\tUNIPROT_END\tPROVENANCE")
    if cath:
        cath_log_file = os.path.join(args.workDir, 'emdb_cath.log')
        cath_log = setup_logger('cath_logger', cath_log_file)
        cath_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tCATH_ID\tSTART\tEND\tUNIPROT_START\tUNIPROT_END\tPROVENANCE")
    if scop:
        scop_log_file = os.path.join(args.workDir, 'emdb_scop.log')
        scop_log = setup_logger('scop_logger', scop_log_file)
        scop_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tSCOP_ID\tSTART\tEND\tUNIPROT_START\tUNIPROT_END\tPROVENANCE")
    if scop2:
        scop2_log_file = os.path.join(args.workDir, 'emdb_scop2.log')
        scop2_log = setup_logger('scop2_logger', scop2_log_file)
        scop2_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tSCOP2_ID\tSTART\tEND\tUNIPROT_START\tUNIPROT_END\tPROVENANCE")
    if scop2B:
        scop2B_log_file = os.path.join(args.workDir, 'emdb_scop2B.log')
        scop2B_log = setup_logger('scop2B_logger', scop2B_log_file)
        scop2B_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tSCOP2B_ID\tSTART\tEND\tUNIPROT_START\tUNIPROT_END\tPROVENANCE")
    if pdbekb:
        pdbekb_log_file = os.path.join(args.workDir, 'emdb_pdbekb.log')
        pdbekb_log = setup_logger('pdbekb_logger', pdbekb_log_file)
        pdbekb_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tPDBeKB_ID\tPROVENANCE")
    if rfam:
        rfam_log_file = os.path.join(args.workDir, 'emdb_rfam.log')
        rfam_log = setup_logger('rfam_logger', rfam_log_file)
        rfam_log.info("EMDB_ID\tEMDB_SAMPLE_ID\tEMDB_SAMPLE_NAME\tNUMBER_OF_COPIES\tRFAM_ACCESSION\tRFAM_ID\tPROVENANCE")

    if uniprot:
        uniprot_dictionary, uniprot_with_models = generate_unp_dictionary(uniprot_tab)
    if component:
        chembl_map, chebi_map, drugbank_map = parseCCD(components_cif)
    if rfam:
        rfam_dictionary = generate_rfam_dictionary(args.workDir)

    pubmed_dict = generate_pubmed_dictionary(args.workDir) if pmc else {}

    if input_json:
        files = read_json(input_json, args.headerDir)
    else:
        files = glob(os.path.join(args.headerDir, 'EMD-*'))

    Parallel(n_jobs=args.threads)(delayed(run)(file) for file in files)
