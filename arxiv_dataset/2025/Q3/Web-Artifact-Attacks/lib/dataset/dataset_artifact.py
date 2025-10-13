import sys
sys.path.append("./")
from PIL import Image
import numpy as np
import random

class ArtifactDataset():

	def __init__(self, args = None, split="test"):
		self.args = args
		self.split = split
		self.transparency = args.transparency
		self.factor_shrink = args.factor_shrink

	def get_lvlm_prompt(self): 
		return self.lvlm_prompt

	def get_prompts(self): 
		prompts = []
		for template in self.get_templates(): 
			prompts.append([template.format(concept).strip() for concept in self.unique_labels])
		
		return prompts, self.unique_labels

	def get_rep_samples(self, num_min = 10, selected_classes = None): 
		to_get_indices = [] 
		np.random.seed(0)
		to_labels = self.unique_labels if selected_classes is None else selected_classes
		for label_ in to_labels:	
			indices = [i for i, label in enumerate(self.labels) if label == label_]
			indices = random.sample(indices, min(num_min, len(indices)))
			to_get_indices.extend(indices)

		self.filenames = [self.filenames[i] for i in to_get_indices]
		self.labels = [self.labels[i] for i in to_get_indices]

	def get_distro(self): 
		counts = {label:0 for label in self.unique_labels}
		for label in self.labels:
			counts[label] += 1
		return counts

	def sample_random(self, p):
		np.random.seed(0)
		idx = np.random.choice(len(self.filenames), int(p*len(self.filenames)), replace=False)
		self.filenames = [self.filenames[i] for i in idx]
		self.build_labels()


	def set_logos_filenames(self, logos=[None]): 
		filenames_ = [] 
		for logo in logos:
			for filename, _ in self.filenames: 
				filenames_.append((filename, logo))
		self.filenames = filenames_
		self.build_labels()

	def load_attack_file(self, paste_attack_file): 
		
		data_new = []
		for past_attack_f, location, logos_type in paste_attack_file:
			if "logos" in logos_type:

				img = Image.open(past_attack_f).convert("RGBA")
				if self.args.logos_mode == "Concept":
					image_array = np.array(img)
					offwhite_condition = (image_array[:, :, :3] > 200).all(axis=2)
					image_array[offwhite_condition] = [255, 255, 255, 0]
					img = Image.fromarray(image_array)

			elif logos_type in ["texts", "typo"]:
				img = Image.open(past_attack_f).convert("RGBA")
				bbox = img.getbbox()
				img = img.crop(bbox)

			else: 
				raise ValueError(f"Invalid type {logos_type}")

			data_new.append((img, location))

		return data_new

	def border_points_for_logo(self, width, height, logo_width, logo_height):
		# Define the adjusted border points
		points = {
			"top_left": (0, 0),
			"top_right": (width - logo_width, 0),
			"bottom_left": (0, height - logo_height),
			"bottom_right": (width - logo_width, height - logo_height),
			"top_middle": ((width - logo_width) // 2, 0),
			"left_middle": (0, (height - logo_height) // 2),
			"bottom_middle": ((width - logo_width) // 2, height - logo_height),
			"right_middle": (width - logo_width, (height - logo_height) // 2)
		}
		
		return points


	def random_border_points_for_logo(self, img_width, img_height, logo_width, logo_height):

		border_percentage = 0.5

		# Calculate the border area dimensions
		border_width = int(img_width * border_percentage)
		border_height = int(img_height * border_percentage)

		# Define the possible area for placing the logo
		possible_positions = [
			(random.randint(0, border_width - logo_width), random.randint(0, img_height - logo_height)),  # Left border
			(random.randint(img_width - border_width, img_width - logo_width), random.randint(0, img_height - logo_height)),  # Right border
			(random.randint(0, img_width - logo_width), random.randint(0, border_height - logo_height)),  # Top border
			(random.randint(0, img_width - logo_width), random.randint(img_height - border_height, img_height - logo_height))  # Bottom border
		]

		return random.choice(possible_positions)

	def past_attack(self, img, past_attack_fs):

		image = img.convert('RGBA')

		area_img = image.size[0] * image.size[1]
		area_logo = area_img // self.factor_shrink
		logo_h_w = int(area_logo ** 0.5)

		transparency = int(self.transparency * 255)

		watermarked_image = Image.new("RGBA", image.size)
		watermarked_image.paste(image, (0, 0))

		for past_attack_f, location in past_attack_fs: 

			watermark = past_attack_f.resize((logo_h_w, logo_h_w))

			img_width, img_height = image.size
			logo_width, logo_height = watermark.size

			if location == "random":
				# img_w, img_h = self.random_border_points_for_logo(img_width, img_height, logo_width, logo_height)
				img_w, img_h = self.border_points_for_logo(img_width, img_height, logo_width, logo_height)[random.choice(["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"])]
			else: 
				img_w, img_h = self.border_points_for_logo(img_width, img_height, logo_width, logo_height)[location]

			watermark = watermark.copy()
			alpha = watermark.split()[3]  # Extract the alpha channel
			alpha = alpha.point(lambda p: p * (transparency / 255))
			watermark.putalpha(alpha)

			# Create a new image to overlay the watermark
			watermarked_image.paste(watermark, (img_w, img_h), mask=watermark)  # Use the watermark as a mask

		return watermarked_image.convert("RGB")

	def __len__(self):
		return len(self.filenames)


	def __getitem__(self, index):
		
		img_path, logo = self.filenames[index]
		
		img_original = Image.open(img_path).convert("RGB").resize((self.image_resize, self.image_resize))
		logo_img = self.load_attack_file(logo) if logo[0][0] is not None else None

		label = self.labels[index]

		img = self.past_attack(img_original, logo_img) if logo_img is not None else img_original
		logo_path = "None" if logo[0][0] is None else logo[0][0]

		to_return = { 
			"idx": index,

			"img_path": img_path.split("/")[-1],
			"logo_path": logo_path, 
			"img_path_full": img_path,

			"label": label,	

			"images":  np.array(img),
		}

		return to_return 
