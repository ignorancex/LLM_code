import lxml.etree as ET
from models import Protein, Supramolecule, Ligand, Model, Weight, Citation, GO, Sample, Interpro, Pfam, Author, Rfam
import re

class XMLParser:
	"""
	Parse the XML files and store the objects
	"""

	def __init__(self, file):
		self.xml_file = file
		self.emdb_id = ""
		self.proteins = []
		self.supramolecules = []
		self.ligands = []
		self.models = []
		self.rfams = []
		self.citation = None
		self.overall_mw = 0.0
		self.read_xml()

	def get_mw(self, sample):
		if sample.xpath('.//molecular_weight/experimental'):
			return float(sample.find('molecular_weight/experimental').text)
		elif sample.xpath('.//molecular_weight/theoretical'):
			return float(sample.find('molecular_weight/theoretical').text)
		return None

	def get_n_copies(self, sample):
		try:
			number_copies = int(sample.find('number_of_copies').text)
		except AttributeError:
			number_copies = 1

		return number_copies

	def get_multiplier(self, node):
		multiplier = node.copies
		stack = node.parent

		while(len(stack) > 0):
			current_node = stack.pop()
			multiplier *= current_node.copies if current_node.copies else 1
			for parent in current_node.parent:
				stack.append(parent)
		return multiplier

	def sum_mw(self, samples, start_nodes):
		stack = []
		single_mw_list = []
		overall_mw = 0.0
		nodes_counted = set()

		for start_node_id in start_nodes:
			start_node = samples[start_node_id]
			if start_node.mw:
				if (start_node_id not in nodes_counted):
					multiplier = self.get_multiplier(start_node)
					try:
						overall_mw += (start_node.mw*multiplier)
					except TypeError:
						overall_mw += 0.0
					nodes_counted.add(start_node_id)
			else:
				stack.append(start_node)

		while(len(stack) > 0):
			current_node = stack.pop()

			if current_node.mw:
				if (current_node.id not in nodes_counted):
					multiplier = self.get_multiplier(current_node)
					try:
						overall_mw += (current_node.mw*multiplier)
					except TypeError:
						overall_mw += 0.0
					nodes_counted.add(current_node.id)
			else:
				for child in current_node.children:
					child_obj = samples[child.id]
					stack.append(child)
		return overall_mw

	def read_xml(self):
		with open(self.xml_file, 'r') as filexml:
			tree = ET.parse(filexml)
			root = tree.getroot()
			a = root.attrib
			self.emdb_id = a.get('emdb_id')
			protein_cpx = {} #Macromolecule -> Supramolecule

			# Iterate over models
			models = root.xpath(".//pdb_list/pdb_reference/pdb_id/text()")
			for pdb_id in models:
				pdb_id = pdb_id.lower()
				model = Model(self.emdb_id, pdb_id)
				self.models.append(model)

			# Iterate over complexes
			complexes = root.xpath(".//complex_supramolecule")
			for complex_tag in complexes:
				complex_id = complex_tag.attrib['supramolecule_id']
				supramolecule = Supramolecule(self.emdb_id, complex_id)
				supramolecule.id = f"supra_{complex_id}" #TODO: Is this supra_ id used anywhere?
				supramolecule.type = "supra" #TODO: Where is it being used?
				complex_name = complex_tag.find('name').text.replace('\t', ' ').strip()
				supramolecule.name = f"{complex_name}_{complex_id}"
				self.supramolecules.append(supramolecule)

				for macromolecule_id in complex_tag.xpath("macromolecule_list/macromolecule/macromolecule_id/text()"):
					if macromolecule_id in protein_cpx:
						protein_cpx[macromolecule_id].add(complex_id)
					else:
						protein_cpx[macromolecule_id] = set(complex_id)

			# Iterate over proteins and peptides
			proteins = root.xpath(".//protein_or_peptide")
			for protein_tag in proteins:
				sample_id = protein_tag.attrib['macromolecule_id']
				protein = Protein(self.emdb_id, sample_id)
				protein.pdb = self.models
				protein.sample_name = protein_tag.find('name').text.replace('\t', ' ').strip()
				if sample_id in protein_cpx:
					protein.sample_complexes = list(protein_cpx[sample_id])
				if protein_tag.find('number_of_copies') is not None:
					protein.sample_copies = protein_tag.find('number_of_copies').text
				else:
					protein.sample_copies = "1"

				organism_ncbi = protein_tag.xpath("natural_source/organism/@ncbi")
				if organism_ncbi:
					protein.sample_organism = organism_ncbi[0]

				uniprot_id = ""
				go_id = ""
				ipr_id = ""
				pfam_id = ""
				for xref in protein_tag.xpath("sequence/external_references"):
					if xref.attrib['type'] == 'UNIPROTKB':
						uniprot_id = xref.text
					elif xref.attrib['type'] == 'GO':
						go_id = xref.text
					elif xref.attrib['type'] == 'INTERPRO':
						ipr_id = xref.text
					elif xref.attrib['type'] == 'PFAM':
						pfam_id = xref.text
				if uniprot_id:
					protein.uniprot_id = uniprot_id
					protein.provenance = "EMDB"
					if go_id:
						go = GO()
						go.add_from_author(go_id, uniprot_id)
						if go.id and go.namespace and go.type:
							protein.go.add(go)
					if ipr_id:
						ipr = Interpro()
						ipr.add_from_author(ipr_id, uniprot_id)
						if ipr.id and ipr.namespace:
							protein.interpro.add(ipr)
					if pfam_id:
						pfam = Pfam()
						pfam.add_from_author(pfam_id, uniprot_id)
						if pfam.id:
							protein.pfam.add(pfam)

				sequence = protein_tag.xpath("sequence/string/text()")
				if sequence:
					sequence = re.sub(r'\(.*?\)', 'X', sequence[0])
					sequence = sequence.replace("\n", "")
					protein.sequence = sequence
				self.proteins.append(protein)

			# Iterate over RNAs
			rnas = root.xpath(".//rna")
			for rna_tag in rnas:
				sample_id = rna_tag.attrib.get('macromolecule_id')
				rfam = Rfam(self.emdb_id, sample_id)
				sampleName = rna_tag.find("name")
				if sampleName is not None:
					rfam.sample_name = sampleName.text
				numberCopies = rna_tag.find("number_of_copies")
				if numberCopies is not None:
					rfam.num_copies = numberCopies.text
				rfam.pdb_id = self.models
				self.rfams.append(rfam)

			# Iterate over Ligands
			ligands = root.xpath(".//ligand")
			for ligand_tag in ligands:
				ligand_id = ligand_tag.attrib['macromolecule_id']
				ligand = Ligand(self.emdb_id, ligand_id)
				HET = ligand_tag.find('formula')
				if HET is not None:
					ligand.HET = HET.text
					ligand_name = ligand_tag.find('name')
					if ligand_name is not None:
						ligand.name = ligand_name.text
					ligand_copies = ligand_tag.find('number_of_copies')
					if ligand_copies is not None:
						ligand.copies = ligand_copies.text

					for xref in ligand_tag.iter('external_references'):
						if xref.attrib['type'] == 'CHEMBL':
							ligand.chembl_id = xref.text
							ligand.provenance_chembl = "EMDB"
						if xref.attrib['type'] == 'CHEBI':
							ligand.chebi_id = xref.text
							ligand.provenance_chebi = "EMDB"
						if xref.attrib['type'] == 'DRUGBANK':
							ligand.drugbank_id = xref.text
							ligand.provenance_drugbank = "EMDB"
					self.ligands.append(ligand)

			# Iterate over primary citation
			primary_citation_list = root.xpath('.//primary_citation/journal_citation')
			if primary_citation_list:
				journal_citation = primary_citation_list.pop()
				citation = Citation(self.emdb_id)
				for author_tag in journal_citation.iter('author'):
					author = Author(author_tag.text, int(author_tag.attrib['order']))
					if 'ORCID' in author_tag.attrib:
						author.orcid = author_tag.attrib['ORCID']
					citation.authors.append(author)
				citation.title = journal_citation.find('title').text.strip()
				citation.published = True if journal_citation.get("published") == "true" else False
				citation_refs = journal_citation.xpath("external_references")
				for xref in citation_refs:
					ref_value = xref.text
					ref_db = xref.get('type')
					if ref_db == "PUBMED":
						citation.pmedid = ref_value
						citation.provenance_pm = "EMDB"
					elif ref_db == "DOI":
						doi = ref_value.split(":")[1]
						citation.doi = doi
						citation.provenance_doi = "EMDB"
					elif ref_db == "ISSN":
						citation.issn = ref_value
						citation.provenance_pm = "EMDB"
				self.citation = citation

			#MW calculation
			sample_dic = {}
			start_nodes = set()
			macromolecules = set()
			sample = root.find('sample')
			supramolecule_list = root.xpath('.//sample/supramolecule_list/*')
			macromolecule_list = root.xpath('.//sample/macromolecule_list/*')

			for sample in macromolecule_list:
				sample_id = 'm' + sample.attrib['macromolecule_id']
				
				number_copies = self.get_n_copies(sample)
				mw = self.get_mw(sample)

				sample_obj = Sample(sample_id,mw,number_copies)
				sample_dic[sample_id] = sample_obj

				if mw:
					macromolecules.add(sample_id)

			for sample in supramolecule_list:
				sample_id = 's' + sample.attrib['supramolecule_id']
				try:
					parent = sample.find('parent').text
				except AttributeError:
					parent = '0'

				number_copies = self.get_n_copies(sample)
				mw = self.get_mw(sample)

				sample_obj = Sample(sample_id,mw,number_copies)
				
				if parent == '0':
					start_nodes.add(sample_id)
				else:
					parent_id = 's' + parent
					if parent_id in sample_dic:
						parent_obj = sample_dic[parent_id]
						sample_obj.add_parent(parent_obj)
						parent_obj.add_child(sample_obj)

				if sample.xpath('.//macromolecule_list'):
					child_list = sample.xpath('.//macromolecule_list/macromolecule/macromolecule_id/text()')
					for child in child_list:
						child_id = "m" + child
						if child_id in sample_dic:
							child_obj = sample_dic[child_id]
							child_obj.add_parent(sample_obj)
							sample_obj.add_child(child_obj)
			
				sample_dic[sample_id] = sample_obj

			for molecule_id in macromolecules:
				molecule = sample_dic[molecule_id]
				if len(molecule.parent) == 0:
					start_nodes.add(molecule_id)

			self.overall_mw = self.sum_mw(sample_dic, start_nodes)
