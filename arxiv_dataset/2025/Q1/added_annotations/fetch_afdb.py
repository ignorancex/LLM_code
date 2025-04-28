import argparse, configparser, os
from pathlib import Path

def get_afdb_ids(alphafold_ftp):
    alphafold_ids = set()
    with open(alphafold_ftp) as f:
        for line in f:
            id = line.split(',')[0]
            alphafold_ids.add(id)
    return alphafold_ids

if __name__ == "__main__":
    prog = "ALPHAFOLD DB(EMICSS)"
    usage = """
            EMICSS for AlphaFold DB
            Example:
            python fetch_afdb.py -w '[{"/path/to/working/folder"}]'
          """

    parser = argparse.ArgumentParser(prog=prog, usage=usage, add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument('-w', '--workDir', type=Path, help="Main working directory path .")
    args = parser.parse_args()

    work_dir = args.workDir
    uniprot_file = os.path.join(work_dir, 'emdb_uniprot.log')
    afdb_map_file = os.path.join(work_dir, 'emdb_alphafold.log')

    #Get config variables:
    config = configparser.ConfigParser()
    env_file = os.path.join(Path(__file__).parent.absolute(), "config.ini")
    config.read(env_file)
    afdb_file = config.get("file_paths", "alphafold_ftp")

    afdb = get_afdb_ids(afdb_file)

    uniprot_reader = open(uniprot_file, 'r')
    next(uniprot_reader) #Skip header
    afdb_writer = open(afdb_map_file, 'w')
    afdb_writer.write("EMDB_ID\tEMDB_SAMPLE_ID\tALPHAFOLDDB_ID\tPROVENANCE\n")

    for line in uniprot_reader:
        line = line.strip()
        row = line.split("\t")
        if len(row) > 0:
            emdb_id = row[0]
            sample_id = row[1]
            unp_id = row[5]
            if unp_id in afdb:
                afdb_writer.write(f"{emdb_id}\t{sample_id}\t{unp_id}\tAlphaFold DB\n")
