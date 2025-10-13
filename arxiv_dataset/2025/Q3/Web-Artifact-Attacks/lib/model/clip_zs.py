import os 
import torch
from tqdm import tqdm
from PIL import Image
import random
import numpy as np 
import open_clip 
import torch.nn as nn

class ClipZeroShot(nn.Module):

	def	__init__(self, args, prompts):
		super(ClipZeroShot, self).__init__()

		self.args = args
		self.model, _, self.preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
		self.tokenizer = open_clip.get_tokenizer(args.model_name)
		self.model = self.model.cuda()
		self.prompts = prompts
		
	def process_images(self, images): 
		processed_images = [] 
		for image_set in images: 
			processed_images.append(torch.cat([self.preprocess(img).unsqueeze(0) for img in image_set], dim=0).unsqueeze(0))
		return processed_images

	def forward(self, batch):

		images = batch["images"]
		images = self.process_images(images)
		images = torch.cat(images, dim=0).cuda()
		
		image_features = []
		for num_img in range(images.shape[1]):
			img = images[:, num_img]
			image_feature = self.model.encode_image(img)
			image_feature /= image_feature.norm(dim=-1, keepdim=True)
			image_feature = image_feature.unsqueeze(1)
			image_features.append(image_feature)

		image_features = torch.cat(image_features, dim=1)

		per_template_scores = [] 
		for prompt_template in self.prompts:
			with torch.no_grad(), torch.cuda.amp.autocast():

				text = self.tokenizer(prompt_template).cuda()
				text_features = self.model.encode_text(text)
				text_features /= text_features.norm(dim=-1, keepdim=True)

				text_probs = [] 
				for num_img in range(image_features.shape[1]):
					text_probs.append((100.0 * image_features[:, num_img] @ text_features.T).unsqueeze(1))
				
				text_probs = torch.cat(text_probs, dim=1)
				text_probs = torch.mean(text_probs, dim=1)

			per_template_scores.append(text_probs.unsqueeze(1))

		per_template_scores = torch.cat(per_template_scores, dim=1)
		per_image_scores = torch.mean(per_template_scores, dim=1)
		return per_image_scores        