import os
import lxml.etree as ET

class StructureMapping:
	"""
	Map PDB ids and they overall molecular weight
	"""
	def __init__(self, models, assembly_ftp):
		self.models = models
		self.assembly_ftp = assembly_ftp

	def execute(self):
		for model in self.models:
			model = self.worker(model)
		return self.models

	def worker(self, model):
		try:
			assembly_file = os.path.join(self.assembly_ftp, "%s/%s/%s-assembly.xml" % (model.pdb_id[1:3],model.pdb_id,model.pdb_id))
			mw, order = self.parse_assembly(assembly_file)
			if mw:
				model.molecular_weight = round(mw, 3)
				model.assembly = order
				return model
		except Exception as e:
			print(e)
		return None

	def parse_assembly(self, file):
		with open(file, 'r') as fr:
			tree = ET.parse(fr)
			root = tree.getroot()
			assemblies = tree.xpath("//assembly")
			for assembly in assemblies:
				if assembly.attrib['prefered'] == 'True':
					mw = round(float(assembly.attrib['molecular_weight']), 3)
					order = int(assembly.attrib['order'])
					return mw,order
			return None, None

	def export_tsv(self, model_logger):
		for model in self.models:
			model_logger.info(str(model))