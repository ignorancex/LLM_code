import pathlib
import argparse
import sys
import os
import mysql.connector
from datetime import datetime
import configparser
from pathlib import Path

latest_data_path = sys.argv[1]
previous_data_path = sys.argv[2]

#Get config variables:
config = configparser.ConfigParser()
env_file = os.path.join(Path(__file__).parent.absolute(), "config.ini")
config.read(env_file)

resources_2_column = {
	"ALPHAFOLD": "afdb",
	"CATH": "cath",
	"CHEBI": "chebi",
	"CHEMBL": "chembl",
	"CPX": "cpx",
	"DRUGBANK": "drugbank",
	"EMPIAR": "empiar",
	"GO": "go",
	"INTERPRO": "interpro",
	"MODEL": "pdbe",
	"PDBEKB": "pdbekb",
	"PFAM": "pfam",
	"PUBMED": "pubmed",
	"SCOP": "scop",
	"SCOP2": "scop2",
	"UNIPROT": "uniprot"
}

resources_count = {
	"orcid": 0,
	"exp_atlas": 0,
	"singlec_atlas": 0,
	"afdb": 0,
	"cath": 0,
	"chebi": 0,
	"chembl": 0,
	"cpx": 0,
	"drugbank": 0,
	"empiar": 0,
	"go": 0,
	"interpro": 0,
	"pdbe": 0,
	"pdbekb": 0,
	"pfam": 0,
	"pubmed": 0,
	"scop": 0,
	"scop2": 0,
	"uniprot": 0
}

mydb = mysql.connector.connect(
  host=config["db"]["HOST"],
  user=config["db"]["USER"],
  password=config["db"]["PWD"],
  port=config["db"]["PORT"],
  database=config["db"]["NAME"]
)

def count_unique(filename):
	entries = set()
	with open(filename, 'r') as fr:
		for line in fr:
			emdb_id = line.split("\t")[0]
			entries.add(emdb_id)
	return len(entries)-1 #Subtract the index


mycursor = mydb.cursor()
message = "Resource    Number of entries (current)    Number of entries (previous)    Difference\n"

for latest_file_path in pathlib.Path(latest_data_path).glob('*.log'):
	filename = latest_file_path.parts[-1]
	resource = filename.split("_")[-1].split(".")[0].upper()
	count_latest = count_unique(str(latest_file_path))
	if resource in resources_2_column:
		column_name = resources_2_column[resource]
		resources_count[column_name] = count_latest
	previous_file_path = previous_data_path + filename.replace('log','tsv')
	if pathlib.Path(previous_file_path).is_file():
		count_previous = count_unique(str(previous_file_path))
		diff = count_latest-count_previous
		diff_str = f"+{diff}" if diff > 0 else str(diff)
		message += f"{resource}    {count_latest}    {count_previous}    {diff_str}\n"

#Insert into DB
today = datetime.today().strftime('%Y-%m-%d')
cols = "date"
values = f'"{today}"'
for col, value in resources_count.items():
	cols += f", {col}"
	values += f", {value}"

sql = f"INSERT INTO emdb_emicssstats ({cols}) VALUES ({values})"
mycursor.execute(sql)
mydb.commit()

os.system(f'mail -s "Summary of weekly annotations workflow" pdb_em@ebi.ac.uk <<< "{message}"')
