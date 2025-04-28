import argparse
from pathlib import Path
from EMICSS.DBVersion import Version 
from EMICSS.EmicssGenerator import Parser, EmicssXML

if __name__ == "__main__":
    prog = "Write EMICSS annotations to XML files"
    usage = """
            EMICSS to EMICSS-XML 
            Example:
            python write_xml.py -w '[{"/path/to/working/folder"}]'
          """

    parser = argparse.ArgumentParser(prog=prog, usage=usage, add_help=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument('-w', '--workDir', type=Path, help="Main working directory path .")
    args = parser.parse_args()
    workDir = args.workDir

    db_version = Version()
    generator = Parser(workDir)
    for emdb_id in generator.emdb_ids:
        print(f"Writing {emdb_id} XML")
        packed_data = generator.get_packed_data(emdb_id)
        emicss = EmicssXML(emdb_id, db_version, packed_data)
        emicss.write_xml(workDir)
