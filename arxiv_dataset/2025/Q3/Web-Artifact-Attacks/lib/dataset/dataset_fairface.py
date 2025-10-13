import sys
sys.path.append("./")
import utils
import torch
import os
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
import random
from transformers import Owlv2Processor
from lib.dataset.dataset_artifact import ArtifactDataset

class FairFace(ArtifactDataset):
	def __init__(self, args = None, split="val"):
		ArtifactDataset.__init__(self, args, split)
		
		self.root = "/usr4/cs505/mqraitem/ivc-ml/datasets/FairFace"

		self.build_data()
		self.build_labels()		
		self.image_resize = 224

	def get_templates(self):
		raw_data = pd.read_csv("prompts/fairface.csv")
		templates = raw_data["template"].dropna().tolist()[:-1]
		return templates

	def categorize_age_labels(self, age_label):
		if age_label in ['0-2', '3-9']:
			return 'Child'
		if age_label in ['10-19']:
			return 'Teenager'
		elif age_label in ['40-49', '50-59', '20-29', '30-39']:
			return 'Adult'
		elif age_label in ['60-69', 'more than 70']:
			return 'Senior'
		else:
			return 'Unknown'  # For unexpected labels

	def build_labels(self):
		df = pd.read_csv(os.path.join(self.root, f"fairface_label_{self.split}.csv"))
		df.set_index('file', inplace=True)
		df['age_category'] = df['age'].map(self.categorize_age_labels)

		index_filenames = [f"{self.split}/{filename.split('/')[-1]}" for filename, _ in self.filenames]

		if self.args.concept == "age":
			self.labels = list(df.loc[index_filenames, 'age_category'].values)
			self.unique_labels = list(set(self.labels))
			self.topic = "Pesron age"

		elif self.args.concept == "gender":
			self.labels = list(df.loc[index_filenames, "gender"].values) 
			self.labels =  ["Woman" if gender == "Female" else "Man" for gender in self.labels ]
			self.unique_labels = list(set(self.labels))
			self.topic = "Person gender"

		else: 
			raise ValueError(f"Invalid concept {self.args.concept}")

		self.unique_labels.sort()
		self.lvlm_prompt = "The person in this image: "
		for idx, label in enumerate(self.unique_labels):
			self.lvlm_prompt += f"{idx+1}) {label}, "
		
		self.lvlm_prompt = self.lvlm_prompt[:-2] + "?"
		self.lvlm_prompt += " Answer with the corresponding number only."

	

	def build_data(self):

		self.filenames = os.listdir(os.path.join(self.root, self.split))
		self.filenames = [[os.path.join(self.root, self.split, filename), None] for filename in self.filenames if ".jpg" in filename]


	def get_unqiue_labels(self):
		return self.unique_gender + self.unique_age + self.unique_race

	def get_all_labels(self, choice): 
		if choice == "age":
			return self.age_labels
		elif choice == "race":
			return self.race_labels
		elif choice == "gender":
			return self.gender_labels
		else: 
			raise ValueError(f"Invalid choice {choice}")



# race = self.race_labels[index]
# gender = self.gender_labels[index]
# age = self.age_labels[index]

# img.save(f"temp_{self.args.logos_type}.jpg")	
# quit()


# self.race_labels = list(df.loc[index_filenames, 'race'].values) 
# self.gender_labels = list(df.loc[index_filenames, 'gender'].values)
# self.age_labels = list(df.loc[index_filenames, 'age_category'].values)

# self.unique_gender = list(set(self.gender_labels))
# self.unique_age = list(set(self.age_labels))
# self.unique_race = list(set(self.race_labels))

# self.unique_gender.sort()
# self.unique_age.sort()
# self.unique_race.sort()

# def get_prompts_all(self): 
# 	all_pairs =  self.unique_gender + self.unique_age + self.unique_race
# 	prompts = utils.get_prompts(utils.get_templates(), all_pairs)
# 	# prompts = utils.get_prompts_dict(clip_prompts, all_pairs)
# 	pairs_to_index = {pair: idx for idx, pair in enumerate(all_pairs)}
# 	return prompts, all_pairs, pairs_to_index


# self.gender_labels = ["Woman" if gender == "Female" else "Man" for gender in self.gender_labels ]

# self.unique_race = list(set(self.race_labels))
# self.unique_gender = list(set(self.gender_labels)) 
# self.unique_age = list(set(self.age_labels))

# self.unique_race.sort()
# self.unique_gender.sort()
# self.unique_age.sort()



# def set_max_samples(self, num): 
# 	#sample num max randomly 
# 	np.random.seed(0)
# 	idx = np.random.choice(len(self.filenames), num, replace=False)
# 	self.filenames = [self.filenames[i] for i in idx]
# 	self.build_labels()


# def get_rep_samples_concept(self, num_min = 10, choice = "age"):

# 	if choice == "age":
# 		pairs = self.unique_age
# 		labels = self.age_labels
# 	elif choice == "gender":
# 		pairs =	self.unique_gender
# 		labels = self.gender_labels
# 	else:
# 		raise ValueError(f"Invalid choice {choice}")

# 	to_get_indices = []
# 	np.random.seed(0)
# 	for pair in pairs:
# 		#get all the indicies with that label
# 		indices = [i for i, label in enumerate(labels) if label == pair]
# 		indices = random.sample(indices, num_min)
# 		to_get_indices.extend(indices)
	
# 	self.filenames = [self.filenames[i] for i in to_get_indices]
# 	self.build_labels()


# def get_distro(self, choice): 
# 	#count how many labels are in each category
# 	if choice == "age":
# 		unique_labels = self.unique_age
# 		labels = self.age_labels
# 	elif choice == "gender":
# 		unique_labels = self.unique_gender 
# 		labels = self.gender_labels 

# 	counts = [0] * len(unique_labels)
# 	for label in labels:
# 		counts[unique_labels.index(label)] = labels.count(label)
	
# 	return counts 



# def limit_to_pair(self, concept, pair): 
# 	indices = [] 
# 	if concept == "age":
# 		labels = self.age_labels
# 	elif concept == "gender":
# 		labels = self.gender_labels
# 	else: 
# 		raise ValueError(f"Invalid concept {concept}")

# 	for idx, label in enumerate(labels):
# 		if label == pair:
# 			indices.append(idx)
	
# 	self.filenames = [self.filenames[i] for i in indices]
# 	self.build_labels()



# def subsample_dataset(self, ratio):
# 	np.random.seed(42)
# 	random_indices = np.random.choice(len(self.filenames), int(len(self.filenames) * ratio), replace=False)
	
# 	self.filenames = [self.filenames[i] for i in random_indices]
# 	self.race_labels = [self.race_labels[i] for i in random_indices]
# 	self.gender_labels = [self.gender_labels[i] for i in random_indices]
# 	self.age_labels = [self.age_labels[i] for i in random_indices]



# layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
# layer.paste(watermark, (img_w, img_h))

# # Create a copy of the layer
# layer2 = layer.copy()

# # Put alpha on the copy
# layer2.putalpha(transparency)
# # merge layers with mask
# layer.paste(layer2, layer)
# result = Image.alpha_composite(image, layer)



# clip_prompts = {
# "Man": [
# 	"man",
# 	"male",
# 	"boy",
# 	"guy",
# 	"gentleman",
# 	"lad",
# 	"male adult",
# 	"young man"
# ],
# "Woman": [
# 	"woman",
# 	"female",
# 	"girl",
# 	"lady",
# 	"gal",
# 	"female adult",
# 	"young woman",
# 	"old woman"
# ],
# "Adult": [
# 	"an adult",
# 	"a grown-up",
# 	"a middle-aged person",
# 	"a working-age individual",
# 	"a legal adult",
# 	"a person in their 30s",
# 	"a person in their 40s",
# 	"a fully grown person"
# ],
# "Teenager": [
# 	"a teenager",
# 	"a teen",
# 	"a high school student",
# 	"a young person in their teens",
# 	"an adolescent",
# 	"a middle school student",
# 	"a young person under 20",
# 	"a person in their late teens"
# ],
# "Senior": [
# 	"a senior",
# 	"an elder",
# 	"an elderly person",
# 	"a retired individual",
# 	"a person of advanced age",
# 	"a senior citizen",
# 	"an older adult",
# 	"a grandparent"
# ],
# "Child": [
# 	"a child",
# 	"a kid",
# 	"a young child",
# 	"a toddler",
# 	"an infant",
# 	"a school-aged person",
# 	"a preteen",
# 	"a person under 12"
# ]
# }
