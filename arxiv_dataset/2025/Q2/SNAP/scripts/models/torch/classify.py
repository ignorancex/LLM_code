import torch
from PIL import Image
import pandas as pd
import argparse
import importlib
import numpy as np
import pickle
import os
import json
from tqdm import tqdm
from torchinfo import summary
import torch.nn.functional as F

def get_imagenet_classes():
	with open("imagenet_classes.txt", "r") as f:
		categories = [s.strip() for s in f.readlines()]
	return categories

class TimmModel:
	def __init__(self, model_name=None, gpu_device='0'):
		self.model_name = model_name
		gpu_devices = gpu_device.split(',')
		if len(gpu_devices) == 1 and int(gpu_devices[0]) < 0:
				self.device = torch.device('cpu')
		else:
			self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

	def load_model(self):
		import timm
		from timm.data import resolve_data_config
		from timm.data.transforms_factory import create_transform
		model = timm.create_model(self.model_name, pretrained=True)
		self.model = model.eval()
		data_config = timm.data.resolve_model_data_config(self.model)
		self.processor = create_transform(**data_config, is_training=False)
		self.labels = get_imagenet_classes()

	def write_pickle(self, data, filename):
		with open(filename, 'wb') as fid:
			pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)

	def predict_list(self, data_path=None, input_path=None, results_path=None):
		
		print(f'Loading model {self.model_name}')
		self.load_model()
		self.model.to(self.device)
		
		print(f'Loading input list from {input_path}...')
		df = pd.read_excel(input_path, engine='openpyxl')
		
		cache_file = os.path.join('cache', os.path.basename(results_path).replace('.xlsx', '.pkl').replace('/', '_'))
		results_dict = []
		if os.path.isfile(cache_file):
			with open(cache_file, 'rb') as fid:
				results_dict = pickle.load(fid)

		processed_images = {x['Path']:0 for x in results_dict}

		num_images = df.shape[0]
		with torch.no_grad():
			num_processed = 0
			for idx, row in tqdm(df.iterrows(), total=num_images):
				if idx >= 18700:
					continue
				if row['Path'] not in processed_images:
					img_path = os.path.join(data_path, row['Path'])
					if os.path.exists(img_path):
						pred_classes = self.predict_img(img_path=img_path)
						results_dict.append({'Path': row['Path'], 'Pred': pred_classes})
						num_processed += 1

				# save results periodically if the script is interrupted
				if (num_processed > 0) and (num_processed % 100 == 0):
					self.write_pickle(results_dict, cache_file)
					#print(f'Saved results to {cache_file}')				


		self.write_pickle(results_dict, cache_file)
		print(f'Saved results to {cache_file}')

		pd.DataFrame.from_dict(results_dict).to_excel(results_path)

	def predict_img(self, img_path=None):
		image = Image.open(img_path)

		labels, predictions = self.process_img(image)

		top5_prob, top5_idx = torch.topk(predictions, k=5)

		top5_prob = [x.item() for x in top5_prob[0]]
		top5_idx = [x.item() for x in top5_idx[0]]

		pred_classes = [(x, labels[x], y) for x, y in zip(top5_idx, top5_prob)]
		return pred_classes

	def process_img(self, image):
		image = self.processor(image).unsqueeze(0).to(self.device)
		logits = self.model(image)
		labels = self.labels
		predictions = torch.nn.functional.softmax(logits, dim=1)
		return labels, predictions


class VitMaeModel(TimmModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)

	def load_model(self):
		# NOTE: to run these models first install timm==0.3.2 (use pip inside the docker)
		# after installing timm, apply a patch from this comment
		# https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842
		
		from argparse import Namespace
		from mae import models_vit
		from mae.util.datasets import build_transform

		checkpoint_urls = {'vit_base_patch16_mae': 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth',
						   'vit_large_patch16_mae': 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth',
						   'vit_huge_patch14_mae': 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth'}

		self.model = models_vit.__dict__[self.model_name.replace('_mae', '')](num_classes=1000, drop_path_rate=0.1, global_pool=True)
		checkpoint = torch.hub.load_state_dict_from_url(checkpoint_urls[self.model_name], map_location='cpu', check_hash=True)
		self.model.load_state_dict(checkpoint['model'])

		args = Namespace(input_size=224)
		is_train = False
		self.processor = build_transform(is_train, args)
		self.labels = get_imagenet_classes()


class OpenClipModel(TimmModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)

	def load_model(self):
		import open_clip
		# from open_clip_train.distributed import init_distributed_device_so
		open_clip.list_pretrained() #to see the full list of trained models
		#print('\n'.join([' '.join(list(x)) for x in open_clip.list_pretrained()]))
		# https://github.com/mlfoundations/open_clip/issues/454
		# model naming conventions
		# s34b - 34b samples seen
		# b88k - batch size 88i
		# laion2b_s34b_b88k
		#self.device = torch.device('cpu') #uncomment this to run the vit-bigg on a cpu
		model_str, data_str = self.model_name.replace('openclip/', '').split('/')
		self.model, _, self.processor = open_clip.create_model_and_transforms(model_str, pretrained=data_str)
		self.model = self.model.to(self.device)
		tokenizer = open_clip.get_tokenizer(model_str)
		self.labels = get_imagenet_classes()
		prompts = [f'a photo of {x}' for x in self.labels]
		self.text = tokenizer(prompts).to(self.device)
		self.text_features = None
		#text_features = self.model.encode_text(self.text)
		#self.text_features /= text_features.norm(dim=-1, keepdim=True)

	def process_img(self, image):
		image = self.processor(image).unsqueeze(0).to(self.device)
		image_features = self.model.encode_image(image)
		image_features /= image_features.norm(dim=-1, keepdim=True)
		#image_features = F.normalize(image_features, dim=-1)

		if self.text_features is None:
			cache = f"cache/{self.model_name.replace('/', '_')}"
			if os.path.isfile(cache):
				with open(cache, 'rb') as fid:
					self.text_features = pickle.load(fid)
			else:
				text_features = self.model.encode_text(self.text)
				text_features /= text_features.norm(dim=-1, keepdim=True)
				with open(cache, 'wb') as fid:
						pickle.dump(text_features, fid)
		
		predictions = (100.0 * image_features.to('cpu') @ self.text_features.T).softmax(dim=-1)
		labels = self.labels
		return labels, predictions

class GoogleViTModel(TimmModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)

	def load_model(self):
		from transformers import ViTImageProcessor, ViTForImageClassification
		self.processor = ViTImageProcessor.from_pretrained(self.model_name)
		self.model = ViTForImageClassification.from_pretrained(self.model_name)
	
	def process_img(self, image):
		image = self.processor(images=image, return_tensors='pt').to(self.device)
		logits = self.model(**image).logits
		predictions = torch.nn.functional.softmax(logits, dim=1)
		labels = self.model.config.id2label
		return labels, predictions

class GoogleSigLIPModel(TimmModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)
	def load_model_7(self):
		from transformers import AutoProcessor, AutoModel
		self.model = AutoModel.from_pretrained(self.model_name)
		self.processor = AutoProcessor.from_pretrained(self.model_name)
		self.labels = get_imagenet_classes()
		self.prompts = [f'a photo of {x}' for x in self.labels]

	def process_img(self, image):
		inputs = self.processor(text=self.prompts, images=image, return_tensors="pt", padding=True).to(self.device)
		outputs = self.model(**inputs)
		logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
		predictions = torch.sigmoid(logits_per_image)
		labels = self.labels
		return labels, predictions

class AppleModel(TimmModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)

	def load_model(self):
		import torch.nn.functional as F
		from open_clip import create_model_from_pretrained, get_tokenizer 
		self.model, self.processor = create_model_from_pretrained(f'hf-hub:{self.model_name}')
		
		tokenizer = get_tokenizer('ViT-H-14')
		self.labels = get_imagenet_classes()
		prompts = [f'a photo of {x}' for x in self.labels]
		self.text = tokenizer(prompts).to(self.device)
		text_features = self.model.encode_text(text)
		self.text_features = F.normalize(text_features, dim=-1)

	def process_img(self, image):
		with torch.no_grad(), torch.cuda.amp.autocast():
			image = self.processor(image).unsqueeze(0).to(self.device)
			image_features = self.model.encode_image(image)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			predictions = torch.sigmoid(image_features @ self.text_features.T * self.model.logit_scale.exp() + self.model.logit_bias)
			labels = self.labels
		return labels, predictions

class OpenAIModel(TimmModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)

	def load_model(self):
			from transformers import CLIPProcessor, CLIPModel
			self.model = CLIPModel.from_pretrained(self.model_name)
			self.processor = CLIPProcessor.from_pretrained(self.model_name)
			self.labels = get_imagenet_classes()
			prompts = [f'a photo of {x}' for x in self.labels]
			#self.text = tokenizer(prompts).to(self.device)

	def process_img(self, image):
		inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True).to(self.device)
		outputs = self.model(**inputs)
		logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
		predictions = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
		labels = self.labels
		return labels, predictions

class MSModel(TimmModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)

	def load_model(self):
		from transformers import AutoModelForImageClassification, AutoImageProcessor
		self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
		self.processor = AutoImageProcessor.from_pretrained(self.model_name)

	def process_img(self, image):
		inputs = self.processor(images=image, return_tensors="pt").to(self.device)
		pixel_values = inputs.pixel_values
		outputs = self.model(pixel_values)
		logits = outputs.logits
		labels = self.model.config.id2label
		predictions = torch.nn.functional.softmax(logits, dim=1)
		return labels, predictions

class InternImageModel(MSModel):
	def __init__(self, model_name=None, gpu_device='0'):
		super().__init__(model_name=model_name, gpu_device=gpu_device)	
	def load_model(self):
		from transformers import AutoModelForImageClassification, CLIPImageProcessor
		self.processor = CLIPImageProcessor.from_pretrained(self.model_name)
		self.model = AutoModelForImageClassification.from_pretrained(self.model_name, trust_remote_code=True)
		with open('imagenet_id2label.json', 'r') as fid:
			labels = json.load(fid)
		self.labels = {int(k):v for k, v in labels['id2label'].items()}

	def process_img(self, image):
		inputs = self.processor(images=image, return_tensors="pt").pixel_values#.to(self.device)
		#pixel_values = inputs.pixel_values
		outputs = self.model(inputs)
		predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
		return self.labels, predictions


'''
To run script
python3  classify_image.py --model_name google/vit-large-patch16-224 --input_path ../../sensor_bias_data.xlsx --results_path ../../results/image_classification/vit-large-patch16-224.xlsx
python classify_image.py --model_name ConvNeXtXLarge --input_path ../../camera/sensor_bias_data.xlsx 
'''


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Run image classification algorithms on an image or a list of images')
	parser.add_argument('--model_name', help='Model to run. Options: VGG16, MobileNet, ConvNeXtXLarge')
	parser.add_argument('--results_path', help='Path to excel file with results')
	parser.add_argument('--gpu_device', type=str, default='0', help='gpu device number')

	args = parser.parse_args()
	print(args)

	if 'timm/' in args.model_name:
		im_class = TimmModel(model_name=args.model_name, gpu_device=args.gpu_device)
	elif args.model_name in ['vit_large_patch16_mae', 'vit_base_patch16_mae', 'vit_huge_patch14_mae']:
		im_class = VitMaeModel(model_name=args.model_name, gpu_device=args.gpu_device)
	elif 'openclip/' in args.model_name:
		im_class = OpenClipModel(model_name=args.model_name, gpu_device=args.gpu_device)
	elif 'google/vit' in args.model_name:
		im_class = GoogleViTModel(model_name=args.model_name, gpu_device=args.gpu_device)
	elif 'google/siglip' in args.model_name:
		im_class = GoogleSigLIPModel(model_name=args.model_name, gpu_device=args.gpu_device)
	elif 'apple/' in args.model_name:
		im_class = AppleModel(model_name=args.model_name, gpu_device=args.gpu_device)
	elif 'microsoft/' in args.model_name:
		im_class = MSModel(model_name=args.model_name, gpu_device=args.gpu_device)
	elif 'OpenGVLab/' in args.model_name:
		im_class = InternImageModel(model_name=args.model_name, gpu_device=args.gpu_device)
	else:
		raise NotImplementedError(f'ERROR: {args.model_name} is not implemented!')

	data_path='data'
	input_path = '../../../annotations/SNAP.xlsx'

	im_class.predict_list(data_path=data_path, input_path=input_path, results_path=args.results_path)