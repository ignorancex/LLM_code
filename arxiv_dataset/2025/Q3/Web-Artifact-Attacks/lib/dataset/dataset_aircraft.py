import os
import json
import pandas as pd
from lib.dataset.dataset_artifact import ArtifactDataset

class Aircraft(ArtifactDataset) :
	def __init__(self, args = None, split="test"):
		ArtifactDataset.__init__(self, args, split)


		self.root_dir =  "/usr4/cs505/mqraitem/ivc-ml/datasets/fgvc-aircraft-2013b"
		self.image_resize = 224
		self.topic = "aircraft model"

		self.build_data()
		self.build_labels() 

	def get_templates(self):
		raw_data = pd.read_csv("prompts/aircraft.csv")
		templates = raw_data["template"].dropna().tolist() 
		return templates

	def build_data(self) :
		aircraft_folder = os.path.join(self.root_dir, "data", "images")
		
		split_file = os.path.join(self.root_dir, "data",f"images_{self.split}.txt")
		samples = open(split_file).read().splitlines()

		with open(f'{self.root_dir}/data/images_manufacturer_{self.split}.txt', 'r') as f:
			aircraft_to_family = f.readlines() 
			aircraft_to_family = {x.split(" ")[0]: x.split(" ")[1].strip() for x in aircraft_to_family}


		selected_aircraft_brands = [
			'boeing',       # Major American aircraft manufacturer
			'airbus',       # Major European competitor to Boeing
			'bombardier',   # Canadian aircraft manufacturer
			'embraer',      # Brazilian competitor to Bombardier
			'cessna',       # General aviation (small aircraft)
			'beechcraft',   # Similar to Cessna (general aviation)
			'robin',     # Military and commercial aircraft
			'mcdonnell',    # Former American military aircraft maker (merged into Boeing)
			'ilyushin',     # Russian aircraft manufacturer
			'piper'       # Russian competitor (similar to Ilyushin)
		]


		self.filenames = [] 
		for filename in samples:
			aircraft_family = aircraft_to_family[filename]
			if aircraft_family.lower() in selected_aircraft_brands:
				self.filenames.append([os.path.join(aircraft_folder, f"{filename}.jpg"), None])
		
		
	def build_labels(self): 

		with open(f'{self.root_dir}/data/images_manufacturer_{self.split}.txt', 'r') as f:
			aircraft_to_family = f.readlines() 
			aircraft_to_family = {x.split(" ")[0]: x.split(" ")[1].strip() for x in aircraft_to_family}

		self.labels = [aircraft_to_family[filename.split("/")[-1].split(".")[0]] for filename, _ in self.filenames]
		self.labels = [x.lower() for x in self.labels]

		self.unique_labels = sorted(list(set(self.labels)))

		self.lvlm_prompt = "The aircraft model in this image is: "
		for idx, label in enumerate(self.unique_labels):
			self.lvlm_prompt += f"{idx+1}) {label}, "
		
		self.lvlm_prompt = self.lvlm_prompt[:-2] + "?"
		self.lvlm_prompt += " Answer with the corresponding number only."
