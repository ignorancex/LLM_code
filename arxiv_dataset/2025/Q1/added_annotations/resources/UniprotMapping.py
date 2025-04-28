import os, re, subprocess
import lxml.etree as ET
import logging
from fuzzywuzzy import fuzz

uniprot_api = "www.uniprot.org/uniprot/?query=\"%s\" AND database:(type:pdb %s)&format=tab&limit=100&columns=id,organism-id&sort=score"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(funcName)s:%(message)s')

file_handler = logging.FileHandler('logging_uniprot.log', mode ='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def generate_unp_dictionary(uniprot_tab):
	"""
	Parse the UniProt tab file containing all entries with models and resulting in
	a dictionary of pdb_id -> [(Uniprot_id, protein_names)] and set of uniprot_ids
	"""
	uniprot = {}
	uniprot_with_models = set()
	with open(uniprot_tab, 'r') as fr:
		next(fr) #Skip header
		for line in fr:
			line = line.strip()
			uniprot_id, pdb_list, protein_names = line.split('\t')
			uniprot_with_models.add(uniprot_id)
			pdb_list = pdb_list.split(';')[:-1]
			for pdb_id in pdb_list:
				pdb_id = pdb_id.lower()
				if pdb_id in uniprot:
					uniprot[pdb_id].append((uniprot_id, protein_names))
				else:
					uniprot[pdb_id] = [(uniprot_id, protein_names)]
	return uniprot, uniprot_with_models


class UniprotMapping:
	"""
	Map EMDB protein samples to Uniprot IDs
	"""
	def __init__(self, workDir, proteins, uniprot_dic, blast_db, blastp_bin):
		self.output_dir = workDir
		self.uniprot = uniprot_dic
		self.proteins = proteins
		self.blast_db = blast_db
		self.blastp_bin = blastp_bin

	def export_tsv(self, unp_logger):
		for protein in self.proteins:
			row = protein.get_tsv()
			if row:
				unp_logger.info(row)

	def execute(self):
		for protein in self.proteins:
			protein = self.worker(protein)
		return self.proteins

	def worker(self, protein):
		if not protein.provenance:
			#If contain a model: Uniprot search
			found = False
			if len(protein.pdb) > 0:
				try:
					found = self.query_uniprot(protein)
				except:
					logger.error("%s failed accessing the Uniprot" % protein.emdb_id)
			
			if not found:
				if protein.sequence:
					try:
						protein = self.blastp(protein)
					except:
						logger.error("%s failed in blastp" % protein.emdb_id)

		return protein

	def blastp(self, protein):
		#Create fasta file
		directory = os.path.join(self.output_dir, "fasta")
		if not os.path.exists(directory):
			os.makedirs(directory)
		fasta_file = os.path.join(directory, protein.emdb_id + ".fasta")
		with open(fasta_file, "w") as f:
			f.write(">seq\n%s" % protein.sequence)
		qout = os.path.join(self.output_dir, "fasta", protein.emdb_id + "_" + protein.sample_id + ".xml")
		blastp_command = [self.blastp_bin, "-query", fasta_file, "-db", self.blast_db, "-out", qout, "-outfmt", "5",
						 "-evalue", "1e-40"]
		subprocess.call(blastp_command)
		
		if os.path.isfile(qout):
			uniprot_id = self.extract_uniprot_from_blast(qout, protein.sample_organism)
			if uniprot_id:
				protein.uniprot_id = uniprot_id
				protein.provenance = "UniProt"
		return protein

	def extract_uniprot_from_blast(self, fastafile, ncbi_id):
		"""
		Extracting the UNIPROT ID from BLASTP XML output
		"""
		with open(fastafile, 'r') as outFile:
			tree = ET.parse(outFile)
			root = tree.getroot()
			seq_len = root.find('BlastOutput_query-len').text
			for x in list(root.iter('Hit')):
				Hit_def = x.find('Hit_def').text
				Hit_split = Hit_def.split("OS")
				uniprot = Hit_split[0]
				uniprot_id = uniprot.split("|")[1]
				tax_id = re.findall('OX=(.*)=', Hit_def, re.S)[0].split("GN")[0].strip()
				ident_num = x.find('Hit_len').text
				if tax_id == ncbi_id:
					coverage = int(ident_num) / int(seq_len)
					if coverage < 1.0:
						continue
					return uniprot_id
		return None

	def query_uniprot(self, protein):
		pdb_found = False
		for pdb in protein.pdb:
			pdb_id = pdb.pdb_id
			if pdb_id in self.uniprot:
				uniprot_list = self.uniprot[pdb_id]
				pdb_found = True
				break

		if not pdb_found:
			return False

		best_score = 0
		best_match = ""

		for uniprot_id, uniprot_names in uniprot_list:
			score = fuzz.token_set_ratio(protein.sample_name.lower(), uniprot_names.lower())
			if score > best_score:
				best_score = score
				best_match = uniprot_id
		
		if best_match and best_score > 80:
			protein.uniprot_id = best_match
			protein.provenance = "UniProt"
			return True
		return False

