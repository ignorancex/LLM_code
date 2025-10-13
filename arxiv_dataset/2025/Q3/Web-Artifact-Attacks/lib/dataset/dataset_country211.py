import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import pycountry
import numpy as np 
import random
from lib.dataset.dataset_artifact import ArtifactDataset


class Country211(ArtifactDataset):
	def __init__(self, args = None, split="test"):
		ArtifactDataset.__init__(self, args, split)

		self.root_dir = "/usr4/cs505/mqraitem/ivc-ml/datasets/country211/"
		self.image_resize = 224
		self.topic = "which country"

		self.build_data()
		self.build_labels() 

	def get_templates(self):
		raw_data = pd.read_csv("prompts/country211.csv")
		templates = raw_data["template"].dropna().tolist()[:-1]
		return templates

	def build_data(self):
		self.filenames = [] 
		countries = ["US", "DE", "JP", "IN", "BR", "ZA", "KR", "TR", "AE", "VN"]

		for country_folder in countries:
			samples = [[f"{self.root_dir}/{self.split}/{country_folder}/{filename}", None] for filename in os.listdir(f"{self.root_dir}/{self.split}/{country_folder}")]
			self.filenames.extend(samples)
		
	def build_labels(self):
		self.labels = []

		for filename, _ in self.filenames:
			country_folder = filename.split("/")[-2]
			if country_folder == "XK":
				country_name = "Kosovo"
			else:
				country_name = pycountry.countries.get(alpha_2=country_folder).name
				if ',' in country_name:
					country_name = country_name.split(',')[1].strip() + " " + country_name.split(',')[0].strip()
			self.labels.append(country_name)

		self.unique_labels = list(set(self.labels))
		self.unique_labels.sort()

		self.lvlm_prompt = "The country in this image: "
		for idx, label in enumerate(self.unique_labels):
			self.lvlm_prompt += f"{idx+1}) {label}, "
		
		self.lvlm_prompt = self.lvlm_prompt[:-2] + "?"
		self.lvlm_prompt += " Answer with the corresponding number only."




# def build_data(self):
# 	self.filenames = [] 

# 	countries_folders = os.listdir(os.path.join(self.root_dir, self.split))
# 	self.countries_to_regions = {}
# 	for country in countries_folders:
# 		try: 
# 			sub_region = CountryInfo(country).subregion() 
# 			self.countries_to_regions[country] = sub_region
# 		except:
# 			continue

# 	countries_folders = list(self.countries_to_regions.keys())
# 	for country_folder in countries_folders:
# 		samples = [[f"{self.root_dir}/{self.split}/{country_folder}/{filename}", None] for filename in os.listdir(f"{self.root_dir}/{self.split}/{country_folder}")]
# 		self.filenames.extend(samples)
	
# def build_labels(self):
# 	self.labels = []

# 	for filename, _ in self.filenames:
# 		country_folder = filename.split("/")[-2]
# 		self.labels.append(self.countries_to_regions[country_folder])

# 	self.unique_labels = sorted(list(set(self.labels)))



# countries = os.listdir(os.path.join(self.root_dir, self.split))
# random.seed(0)


# def load_data(self): 
# 	countries_folders = os.listdir(os.path.join(self.root_dir, self.split))
# 	self.filenames = []
# 	self.labels = [] 
# 	for country_folder in countries_folders:
# 		samples = [f"{self.root_dir}/{self.split}/{country_folder}/{filename}" for filename in os.listdir(f"{self.root_dir}/{self.split}/{country_folder}")]
# 		self.filenames.extend(samples)
		
# 		if country_folder == "XK":
# 			country_name = "Kosovo"
# 		else:
# 			country_name = pycountry.countries.get(alpha_2=country_folder).name
# 			if ',' in country_name:
# 				country_name = country_name.split(',')[1].strip() + " " + country_name.split(',')[0].strip()
			
# 		self.labels.extend([country_name] * len(samples))

# 	self.labels = [x.lower() for x in self.labels]
# 	self.all_labels = sorted(list(set(self.labels)))
# 	self.all_labels = [x.lower() for x in self.all_labels]
# 	self.all_labels = [x.lower() for x in self.all_labels]

