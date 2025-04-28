import csv
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict
import argparse
import configparser
import os
from ftplib import FTP
from pathlib import Path
import os


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def read_tsv(file_path: str) -> List[Dict[str, str]]:
    """Read a TSV file and return a list of dictionaries."""
    logging.info(f"Reading TSV file: {file_path}")
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)


def build_xml(data: List[Dict[str, str]]) -> ET.Element:
    """Build the XML structure from the TSV data."""
    logging.info("Building XML structure")
    root = ET.Element("links")
    for row in data:
        emdb_id = row["EMDB_ID"]
        pubmed_id = row["PUBMED_ID"]

        link = ET.SubElement(root, "link", providerId="2057")

        resource = ET.SubElement(link, "resource")
        ET.SubElement(resource, "title").text = emdb_id
        ET.SubElement(resource, "url").text = f"https://www.ebi.ac.uk/emdb/{emdb_id}"

        record = ET.SubElement(link, "record")
        ET.SubElement(record, "source").text = "MED"
        ET.SubElement(record, "id").text = pubmed_id

    return root


def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    return minidom.parseString(ET.tostring(elem, 'utf-8')).toprettyxml(indent="  ")


def write_xml(xml_root: ET.Element, output_file: str) -> None:
    """Write the XML tree to a file with pretty formatting."""
    logging.info(f"Writing XML to file: {output_file}")
    pretty_xml = prettify_xml(xml_root)
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(pretty_xml)


def upload_file_via_ftp(
    server: str,
    username: str,
    password: str,
    local_file_path: str,
    remote_dir: str = ".",
    remote_filename: str = None
) -> None:
    """
    Upload a file to an FTP server. It uses port 21 by default.

    Args:
        server (str): FTP server address.
        username (str): FTP username.
        password (str): FTP password.
        local_file_path (str): Path to the local file to upload.
        remote_dir (str): Remote directory to upload the file to. Default is root.
        remote_filename (str): Optional name to give the uploaded file. Defaults to same as local file.
    """
    if not os.path.exists(local_file_path):
        logging.error(f"Local file does not exist: {local_file_path}")
        return

    remote_filename = remote_filename or os.path.basename(local_file_path)

    try:
        logging.info(f"Connecting to FTP server: {server}")
        with FTP(server) as ftp:
            ftp.login(user=username, passwd=password)
            logging.info(f"Logged in as {username}")

            ftp.cwd(remote_dir)
            logging.info(f"Changed to remote directory: {remote_dir}")

            with open(local_file_path, "rb") as file:
                ftp.storbinary(f"STOR {remote_filename}", file)
                logging.info(f"File uploaded successfully as {remote_filename}")
    except Exception as e:
        logging.error(f"FTP upload failed: {e}")
        return


def main():
    setup_logging()

    input_tsv = "/nfs/ftp/public/databases/em_ebi/emdb_related/emicss/resources/emdb_pubmed.tsv"
    output_xml = "/hps/nobackup/gerard/emdb/annotations/output/EMDB_linkFile_providerID_2057.xml"

    data = read_tsv(input_tsv)
    xml_root = build_xml(data)
    write_xml(xml_root, output_xml)

    logging.info(f"XML file generated: {output_xml}")

    # Upload the XML file to the FTP server
    logging.info("Uploading XML file to FTP server")
    config = configparser.ConfigParser()
    env_file = os.path.join(Path(__file__).parent.absolute(), "config.ini")
    config.read(env_file)
    ftp_server = config.get("epmc_ftp", "server")
    ftp_user = config.get("epmc_ftp", "username")
    ftp_pass = config.get("epmc_ftp", "password")
    ftp_dir = config.get("epmc_ftp", "directory")

    upload_file_via_ftp(
        server=ftp_server,
        username=ftp_user,
        password=ftp_pass,
        local_file_path=output_xml,
        remote_dir=ftp_dir
    )

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
