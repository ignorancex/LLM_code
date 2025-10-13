from pathlib import Path
import os 
import sys
import json
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import argparse
from torchvision.models import get_model, get_model_weights, get_weight
from torchvision.io.image import read_image
import importlib
import gc
from PIL import Image, ImageDraw
import torch
import numpy as np
from torchinfo import summary


class TorchModel:
	def __init__(self, model_name=None, data_path='data', results_path=None):
		self._model_name = model_name
		self._model = None
		self._weights = None
		self._data_path = data_path
		self._data_df = pd.read_excel(f'../../../annotations/SNAP.xlsx')
		self._cache = f'cache/{model_name}.pkl'
		self._results_path = results_path
		self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
		with open('coco_id2label.json', 'r') as fid:
			self._id2label = json.load(fid)
			self._label2id = {v:int(k) for k, v in self._id2label.items()}
		with open('coco_91to80.json', 'r') as fid:
			self._coco_91to80 = json.load(fid)
		with open('detectron2_coco_map.json', 'r') as fid:
			self._det2_coco = json.load(fid)

	def vis_bbox(self, image_path, bbox):
		path = os.path.join(self._data_path, image_path)
		img = Image.open(path)
		img1 = ImageDraw.Draw(img)
		img1.rectangle(bbox, outline='red')
		img.show()

	def load_model(self):
		module = importlib.import_module('torchvision.models.detection')
		model_class = getattr(module, self._model_name.lower())
		self._weights = getattr(module, f'{self._model_name}_Weights').DEFAULT
		self._model = model_class(weights=self._weights).eval()
		self._preprocess = self._weights.transforms()
		self._model.to(self.device)

		# total_params = 0
		# for p in self._model.parameters():
		# 	total_params += p.numel()
		# print(f"Number of parameters: {total_params}")

		# total_params = 0
		# for name, param in self._model.named_parameters():
		# 	print(name, param.numel())
		# 	total_params += param.numel()
		# print(f"Number of parameters: {total_params}")
		# summary(self._model, input_size=(1,3,224,224))

	def get_image_list(self):
		return self._data_df['Path'].tolist()

	def load_results(self):
		if os.path.exists(self._cache):
			with open(self._cache, 'rb') as fid:
				results = pkl.load(fid)
		else:
			results = {}		
		return results

	def cache_results(self):
		with open(self._cache, 'wb') as fid:
			pkl.dump(self.results, fid)

	def load_img(self, path):
		return  read_image(path)

	def process_img(self, img):
		result = self._model([self._preprocess(img).to(self.device)])[0]
		result_cpu = [v.cpu().tolist() for k, v in result.items()]
		return result_cpu

	def predict(self):
		image_list = self.get_image_list()
		self.results = self.load_results()

		self.load_model()
		
		with torch.no_grad():
			updated = False
			for idx, image_path in enumerate(tqdm(image_list, desc=self._model_name)):
				if image_path not in self.results:

					path = os.path.join(self._data_path, image_path)

					if path not in self.results:
						updated = True
						img = self.load_img(path)
						result_cpu = self.process_img(img)
						self.results[image_path] = result_cpu

				if updated and idx % 100 == 0:
					self.cache_results()
					gc.collect()
					updated = False

		self.cache_results()


	def split_result(self, result):
		# for some reason FasterRCNN outputs results not in the same order
		if 'FasterRCNN' in self._model_name:
			confs = result[2]
			classes = result[1]
			boxes = result[0] #.cpu().tolist()
		else:
			confs = result[1]
			classes = result[2]
			boxes = result[0] #.cpu().tolist()
		return confs, classes, boxes

	def get_class_id(self, cl):
		class_id = int(self._coco_91to80[str(cl)])-1 # subtract 1 because we number classes starting with 0
		return class_id

	def get_bbox(self, box):
		bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]] #convert to left-corner-width-height
		return bbox

	def save_predictions_det(self):

		# save predictions in COCO format
		img_path2idx = {v:k for k, v in self._data_df['Path'].to_dict().items()}

		coco_results = []
		coco_result = {'image_id': None, 'category_id': None, 'bbox': None, 'score': None}

		for img_path, result in tqdm(self.results.items(), desc='Saving'):
			if not img_path in img_path2idx:
				continue

			confs, classes, boxes = self.split_result(result)

			for cl, conf, box in zip(classes, confs, boxes):
				if conf >= 0.5:
					coco_result['image_id'] = img_path2idx[img_path]
					coco_result['category_id'] = self.get_class_id(cl) 
					coco_result['bbox'] = self.get_bbox(box)
					coco_result['score'] = conf
					coco_results.append(coco_result.copy())
		with open(self._results_path, 'w') as fid:
			json.dump(coco_results, fid, ensure_ascii=False, indent=2)

	def save_results(self):
		if self._results_path is not None:
				self.save_predictions_det()
 

class HuggingFaceModel(TorchModel):
	def __init__(self, model_name=None, results_path=None):
		super().__init__(model_name=model_name, results_path=results_path)
		self._texts = None
		self._thresh = 0.5

	def load_model(self):

		if self._model_name in ['detr-resnet-50', 'detr-resnet-101']:
			from transformers import DetrForObjectDetection, AutoImageProcessor
			self._model = DetrForObjectDetection.from_pretrained(f'facebook/{self._model_name}')
			self._preprocess = AutoImageProcessor.from_pretrained(f'facebook/{self._model_name}')
			self.id2label = self._model.config.id2label
		elif self._model_name in ['deta-swin-large', 'deta-resnet-50']:
			from transformers import AutoImageProcessor, AutoModelForObjectDetection
			self._preprocess = AutoImageProcessor.from_pretrained(f'jozhang97/{self._model_name}')
			self._model = AutoModelForObjectDetection.from_pretrained(f'jozhang97/{self._model_name}')
			self.id2label = self._model.config.id2label
		elif self._model_name in ['rtdetr_r101vd_coco_o365', 'rtdetr_r101vd', 'rtdetr_r50vd', 'rtdetr_r50vd_coco_o365']:
			from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
			self._preprocess = RTDetrImageProcessor.from_pretrained(f'PekingU/{self._model_name}')
			self._model = RTDetrForObjectDetection.from_pretrained(f'PekingU/{self._model_name}')
			self.id2label = self._model.config.id2label
		elif self._model_name in ['rtdetr_v2_r101vd', 'rtdetr_v2_r50vd']:
			from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
			self._preprocess = RTDetrImageProcessor.from_pretrained(f'PekingU/{self._model_name}')
			self._model = RTDetrV2ForObjectDetection.from_pretrained(f'PekingU/{self._model_name}')
			self.id2label = self._model.config.id2label
		elif self._model_name in ['owlvit-large-patch14']:
			from transformers import OwlViTProcessor, OwlViTForObjectDetection
			self._preprocess = OwlViTProcessor.from_pretrained(f'google/{self._model_name}')
			self._model = OwlViTForObjectDetection.from_pretrained(f'google/{self._model_name}')
			self._texts = [f'{v}' for k,v in self._id2label.items()]
			self._thresh = 0.4
			self.id2label = self._id2label
		elif self._model_name in ['grounding-dino-base']:
			from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
			self._preprocess = AutoProcessor.from_pretrained(f'IDEA-Research/{self._model_name}')
			self._model = AutoModelForZeroShotObjectDetection.from_pretrained(f'IDEA-Research/{self._model_name}')
			self._texts = ' . '.join([f'{v}' for k,v in self._id2label.items()])+'.'
			self.id2label = self._model.config.id2label
		else:
			raise ValueError(f'ERROR: model {self._model_name} does not exist!')

		self._model.to(self.device)
		total_params = 0
		for p in self._model.parameters():
			total_params += p.numel()
		print(f"Number of parameters: {total_params}")

		total_params = 0
		for name, param in self._model.named_parameters():
			#print(name, param.numel())
			total_params += param.numel()
		print(f"Number of parameters: {total_params}")
		#summary(self._model, input_size=(1,3,224,224))
		

	def load_img(self, path):
		return  Image.open(path)

	def process_img(self, img):
		if self._texts is None:
			inputs = self._preprocess(images=img, return_tensors='pt').to(self.device)
		else:
			inputs = self._preprocess(text=self._texts, images=img, return_tensors='pt').to(self.device)
		result = self._model(**inputs)
		if 'grounding' in self._model_name:
			result_cpu = self._preprocess.post_process_grounded_object_detection(result, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=torch.tensor([img.size[::-1]]))
			result_cpu = [result_cpu[0]['scores'].tolist(), result_cpu[0]['labels'], result_cpu[0]['boxes'].tolist()]
		else:
			result_cpu = self._preprocess.post_process_object_detection(result, threshold=self._thresh, target_sizes=torch.tensor([img.size[::-1]]))
			result_cpu = [v.cpu().tolist() for k, v in result_cpu[0].items()]
		return result_cpu

	def split_result(self, result):
		# for some reason FasterRCNN outputs results not in the same order
		confs = result[0]
		classes = result[1]
		boxes = result[2] #.cpu().tolist()
		return confs, classes, boxes

	def get_bbox(self, box):
		bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]] #convert to left-corner-width-height
		return bbox

	def get_class_id(self, cl):
		if isinstance(cl, str):
			class_id = self._label2id.get(cl, -1)
		else:
			num_classes = len(self.id2label)
			if num_classes == 91:
				# these models return classes based on 90 categories, we need to convert them to 80 classes to match our annotations
				class_id = int(self._coco_91to80[str(cl)])-1 # subtract 1 because we number classes starting with 0
			elif num_classes == 80:
				if 'owlvit' in self._model_name or 'rtdetr_v2' in self._model_name:
					class_id = cl
				else:
					class_id = cl-1
			else:
				raise ValueError(f'ERROR: number of classes should be 91 or 80, not {num_classes}!')

		return class_id

# to setup DINO model, extract DINO code into scripts/models/image_classification/Transformers/DINO
# create a DINO/ckpt
# download checkpoint0031_5scale.pth (DINO-R50)
# download checkpoint0027_5scale_swin.pth (DINO-SWIN-L)
# pip3 install addict yapf scipy

# it uses a custom attention implementation, which should be compiled as follows:
# go to DINO/models/dino/ops/
# run sh make.sh or python3 setup.py build install
# However, in docker this does not work, so instead I'm using the torch implementation
# DINO/models/dino/ops/functions/ms_deform_attn_func.py
# modified 
#		output = MSDA.ms_deform_attn_forward(
#		   value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
# to this:
#         output = ms_deform_attn_core_pytorch(
#            value, value_spatial_shapes, sampling_locations, attention_weights)  

class DINOModel(HuggingFaceModel):
	def __init__(self, model_name=None, results_path=None):
		super().__init__(model_name=model_name, results_path=results_path)

		self._model_name = model_name

		self._model_dict = {'dino-r50': {'checkpoint': 'DINO/ckpt/checkpoint0033_4scale.pth', 'config': 'DINO/config/DINO/DINO_4scale.py'},
							'dino-swin-l': {'checkpoint': 'DINO/ckpt/checkpoint0029_4scale_swin.pth', 'config': 'DINO/config/DINO/DINO_4scale_swin.py' }}

		self._img_h = 640
		self._img_w = 960


	def load_model(self):

		sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DINO'))
		

		from main import build_model_main
		from util.slconfig import SLConfig
		from util import box_ops

		import datasets.transforms as T

		model_config_path = self._model_dict[self._model_name]['config']
		model_checkpoint_path = self._model_dict[self._model_name]['checkpoint']

		args = SLConfig.fromfile(model_config_path) 
		args.device = 'cuda' 
		self._model, self._criterion, self._process = build_model_main(args)
		checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
		self._model.load_state_dict(checkpoint['model'])
		self._model.eval()
		self._transform = T.Compose([
				T.RandomResize([800], max_size=1333),
				T.ToTensor(),
				T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

		with open('DINO/util/coco_id2name.json', 'r') as fid:
			self.id2label = json.load(fid)


	def get_bbox(self, box):
		x0, y0, x1, y1 = box
		b = [x0, y0, (x1 - x0), (y1 - y0)]
		bbox = [b[0]*self._img_w, b[1]*self._img_h, b[2]*self._img_w, b[3]*self._img_h]
		return bbox

	def load_img(self, path):
		img = Image.open(path).convert('RGB')
		return img

	def process_img(self, img):

		img, _ = self._transform(img, None)
		result = self._model.cuda()(img[None].to(self.device))
		result = self._process['bbox'](result, torch.Tensor([[1.0, 1.0]]).cuda())[0]
		# filter out boxes with low confidence scores
		keep_idx = result['scores'] >= 0.5
		result_cpu = [v[keep_idx].cpu().tolist() for k, v in result.items()]
		return result_cpu

	def get_class_id(self, cl):
		#class_id = int(self.id2label[str(cl)])
		class_id = self._coco_91to80[str(cl)]-1
		return class_id

class Detectron2Model(HuggingFaceModel):
	def __init__(self, model_name=None, results_path=None):
		super().__init__(model_name=model_name, results_path=results_path)
		self.model_name = model_name
		self.model_url = 'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/'
		self.model_dict = {'mask_rcnn_vitdet_b': ('config/mask_rcnn_vitdet_b_100ep.py', 'mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl'),
						  'mask_rcnn_vitdet_l': ('config/mask_rcnn_vitdet_l_100ep.py', 'mask_rcnn_vitdet_l/f325599698/model_final_6146ed.pkl'),
						  'mask_rcnn_vitdet_h': ('config/mask_rcnn_vitdet_h_75ep.py', 'mask_rcnn_vitdet_h/f329145471/model_final_7224f1.pkl'),
						  'cascade_mask_rcnn_vitdet_b': ('config/cascade_mask_rcnn_vitdet_b_100ep.py', 'cascade_mask_rcnn_vitdet_b/f325358525/model_final_435fa9.pkl'),
						  'cascade_mask_rcnn_vitdet_l': ('config/cascade_mask_rcnn_vitdet_l_100ep.py', 'cascade_mask_rcnn_vitdet_l/f328021305/model_final_1a9f28.pkl'),
						  'cascade_mask_rcnn_vitdet_h': ('config/cascade_mask_rcnn_vitdet_h_75ep.py', 'cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'),
						  'cascade_mask_rcnn_swin_b': ('config/cascade_mask_rcnn_swin_b_in21k_50ep.py', 'cascade_mask_rcnn_swin_b_in21k/f342979038/model_final_246a82.pkl'),
						  'cascade_mask_rcnn_swin_l': ('config/cascade_mask_rcnn_swin_l_in21k_50ep.py', 'cascade_mask_rcnn_swin_l_in21k/f342979186/model_final_7c897e.pkl')
						  }
	def load_model(self):
		from detectron2.checkpoint import DetectionCheckpointer
		from detectron2.config import LazyConfig, instantiate
		from detectron2.data.detection_utils import read_image

		from detectron2.config import get_cfg

		model_config_path = self.model_dict[self.model_name][0]
		cfg = LazyConfig.load(model_config_path)
		self._model = instantiate(cfg.model)
		self._model.to(self.device)
		model_weight_path = self.model_url + self.model_dict[self.model_name][1]
		DetectionCheckpointer(self._model).load(model_weight_path)
		self._model.eval()

	def predict(self):
		from detectron2.data.detection_utils import read_image
		from detectron2.data import MetadataCatalog
		#from detectron2.utils.visualizer import Visualizer

		image_list = self.get_image_list()
		self.results = self.load_results()

		self.load_model()
		with torch.no_grad():
			num_processed = 0
			updated = False
			for image_path in tqdm(image_list, desc=self._model_name):
				if image_path not in self.results:
					path = os.path.join(self._data_path, image_path)
					if path not in self.results:
						img = torch.from_numpy(np.ascontiguousarray(read_image(path, format="BGR")))
						img = img.permute(2, 0, 1).to(self.device)  # HWC -> CHW
						inputs = [{'image': img}]
						predictions = self._model(inputs)[0]['instances']
						result_cpu = predictions.to('cpu')

						self.results[image_path] = [result_cpu.scores.tolist(), result_cpu.pred_classes.tolist(), result_cpu.pred_boxes.tensor.tolist()]

						updated = True

				num_processed += 1

				if num_processed > 0 and num_processed % 100 == 0 and updated:
					self.cache_results()
					gc.collect()
					updated = False
		if updated:
			self.cache_results()	

	def get_class_id(self, cl):
		# detectron2 remaps COCO ids to a continuous range [1, 90]
		# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/coco.py
		# so we need to map it back for evaluation
		coco91id = str(self._det2_coco[str(cl)])
		# these models return classes based on 90 categories, we need to convert them to 80 to match the annotations
		class_id = self._coco_91to80[coco91id]-1
		return class_id


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Run ultralytics algorithms on an image or a list of images')
	parser.add_argument('--model_name', help='Model to run')
	parser.add_argument('--results_path', help='Path to json file with results')

	args = vars(parser.parse_args())
	print(args)

	# NOTE:need to install transformers-4.48.3 torch==2.0.1 and torchvision==0.15.2 inside docker for rtdetr models
	if args['model_name'] in ['rtdetr_v2_r50vd', 'grounding-dino-base', 'owlvit-large-patch14', 'detr-resnet-50', 'detr-resnet-101', 'deta-swin-large', 'deta-resnet-50', 'rtdetr_r50vd', 'rtdetr_r50vd_coco_o365', 'rtdetr_r101vd_coco_o365', 'rtdetr_r101vd', 'rtdetr_v2_r101vd']:
		m = HuggingFaceModel(**args)
	elif 'vitdet' in args['model_name'] or 'cascade' in args['model_name']:
		m = Detectron2Model(**args)
	elif args['model_name'] in ['dino-r50', 'dino-swin-l']:
		m = DINOModel(**args)
	else:
		m = TorchModel(**args)

	m.predict()
	m.save_results()