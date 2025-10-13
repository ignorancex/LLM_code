import sys
import os
import pandas as pd
import torch
from PIL import Image
from lib.dataset.dataset_artifact import ArtifactDataset

class CelebA(ArtifactDataset):
	def __init__(self, args=None, split="val", concept="Blond_Hair"):
		super().__init__(args, split)
		
		self.root = "/usr4/cs505/mqraitem/ivc-ml/datasets/celeba/celeba"
		self.split = split
		self.image_resize = 224
		self.concept = concept
		
		self.build_data()
		self.build_labels()

	def get_templates(self):
		raw_data = pd.read_csv("prompts/celeba.csv")
		templates = raw_data["template"].dropna().tolist()[:-1]
		return templates


	def build_data(self):
		"""Loads image file paths from the dataset."""
		split_file = os.path.join(self.root, "list_eval_partition.txt")
		split_df = pd.read_csv(split_file, delim_whitespace=True, header=None, names=["filename", "split"])
		
		split_map = {"train": 0, "val": 1, "test": 2}
		selected_filenames = split_df[split_df["split"] == split_map[self.split]]["filename"].tolist()
		
		self.filenames = [[os.path.join(self.root, "img_align_celeba", filename), None] 
						  for filename in selected_filenames if filename.endswith(".jpg")]

	def build_labels(self):
		"""Loads and processes labels for the 'Blond_Hair' attribute."""
		attr_file = os.path.join(self.root, "list_attr_celeba.txt")
		df = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)
		
		# Convert -1/1 labels to boolean
		if self.concept == "Blond_Hair":
			df['Blond_Hair'] = df['Blond_Hair'].map(lambda x: "Blonde" if x == 1 else "Brown or Black Hair")
			self.topic = "Hair Color"
		elif self.concept == "Smiling":
			df['Smiling'] = df['Smiling'].map(lambda x: "Smiling" if x == 1 else "Frowning")
			self.topic = "Facial Expression"

		df.index = df.index.astype(str)  # Ensure index is string for matching

		index_filenames = [filename.split('/')[-1] for filename, _ in self.filenames]
		
		self.labels = list(df.loc[index_filenames, self.concept].values)
		self.unique_labels = list(set(self.labels))
		self.unique_labels.sort()

		self.lvlm_prompt = "The person in this image: "
		for idx, label in enumerate(self.unique_labels):
			self.lvlm_prompt += f"{idx+1}) {label}, "
		
		self.lvlm_prompt = self.lvlm_prompt[:-2] + "?"
		self.lvlm_prompt += " Answer with the corresponding number only."

	