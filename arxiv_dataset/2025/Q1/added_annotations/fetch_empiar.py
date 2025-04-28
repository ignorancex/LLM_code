import argparse
import configparser
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import csv

def empiar_mapping(header_dir, empiar_map_file):
    with open(empiar_map_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow(["EMDB_ID", "EMPIAR_ID", "PROVENANCE"])

        for xml_file in Path(header_dir).glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            ns = {'ns': root.tag.split('}')[0].strip('{')} # Extract namespace from the root tag
            emdb_entries = root.findall('.//ns:crossReferences/ns:relatedEMDBEntries/ns:emdbEntry', namespaces=ns)
            empiar_id = "EMPIAR-" + xml_file.stem

            for emdb_entry in emdb_entries:
                csv_writer.writerow([emdb_entry.text, empiar_id, "EMPIAR"])


if __name__ == "__main__":
    prog = "EMPIAR (EMICSS)"
    usage = """
            EMICSS for EMPIAR
            Example:
            python fetch_empiar.py -w '[{"/path/to/working/folder"}]' -f '[{"/path/to/EMPAR/header/folder"}]'
          """

    parser = argparse.ArgumentParser(prog=prog, usage=usage, add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument('-w', '--workDir', type=Path, help="Main working directory path.")
    parser.add_argument('-f', '--headerDir', type=Path, help="Directory path to the EMPIAR header files.")
    args = parser.parse_args()

    work_dir = args.workDir
    header_dir = args.headerDir
    empiar_map_file = os.path.join(work_dir, 'emdb_empiar.log')

    # Get config variables:
    config = configparser.ConfigParser()
    env_file = os.path.join(Path(__file__).parent.absolute(), "config.ini")
    config.read(env_file)

    empiar_mapping(header_dir, empiar_map_file)