from pathlib import Path
import os 
import json
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import argparse
from ultralytics.utils.ops import xywh2ltwh
from ultralytics.engine.results import Masks

class UltralyticsModel:
	def __init__(self, model_name=None, task='detect', input_path=None, results_path=None, data_version='v3'):
		self._task = task
		self._model_name = model_name
		self._model = self.load_model()
		self._data_path = input_path
		self._data_version = data_version
		self._data_df = pd.read_excel(f'../../../annotations/SNAP.xlsx')
		self._cache = f'cache/{model_name}_data_{data_version}.pkl'
		self._results_path = results_path


	def load_model(self):
		model_name = f'{self._model_name}.pt'
		if 'yolo' in self._model_name.lower():
			from ultralytics import YOLO
			model = YOLO(model_name)
		elif 'detr' in self._model_name.lower():
			from ultralytics import RTDETR
			model = RTDETR(model_name)
		elif 'sam' in self._model_name.lower():
			from ultralytics import SAM
			model = SAM(model_name)
		else:
			raise NotImplementedError(f'Cannot load {self._model_name}')

		model.info()

		return model


	@staticmethod
	def masks2segments(masks):
		"""
		Takes a list of masks(n,h,w) and returns a list of segments(n,xy), from
		https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

		Args:
			masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

		Returns:
			segments (List): list of segment masks.
		"""
		segments = []
		for x in masks.astype("uint8"):
			c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
			if c:
				c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
			else:
				c = np.zeros((0, 2))  # no segments found
			segments.append(c.astype("float32"))
		return segments


	def predict(self):
		image_list = [str(x) for x in Path(self._data_path).rglob("*.jpg")]

		if os.path.exists(self._cache):
			with open(self._cache, 'rb') as fid:
				self.results = pkl.load(fid)
		else:
			self.results = {}
			self._model = self.load_model()
		updated = False

		for idx, image_path in enumerate(tqdm(image_list, desc=self._model_name)):
			path = image_path.replace(self._data_path,'').strip('/')
			updated = False
			if path not in self.results:
				result = self._model(image_path,verbose=False)
				for i in range(len(result)):
					#r.show()
					result[i].orig_img = None
					#if result[i].masks is not None:
					#	result[i].masks.data = None
					
				self.results[path] = result
				updated = True

			if updated and idx > 0 and idx % 100 == 0:
				with open(self._cache, 'wb') as fid:
					pkl.dump(self.results, fid)

		if updated:
			with open(self._cache, 'wb') as fid:
				pkl.dump(self.results, fid)

	def save_results(self):
		if self._results_path is not None:
			if self._task == 'classify':
				self.save_predictions_cls()
			elif self._task == 'detect':
				self.save_predictions_det()
			elif self._task == 'segment':
				self.save_predictions_seg()
			else:
				raise ValueError('ERROR: Not implemented!')

	def save_predictions_cls(self):
		# save classification predictions
		results_dict = []
		for img_path, result in self.results.items():
			names = result[0].names
			pred_class_names = [names[x] for x in result[0].probs.top5]
			pred_classes = [x for x in result[0].probs.top5]
			pred_conf = result[0].probs.top5conf
			results_dict.append({'Path': img_path, 'Pred': [(x, y, z) for x, y, z in zip(pred_classes, pred_class_names, pred_conf.tolist())]})
		pd.DataFrame.from_dict(results_dict).to_excel(self._results_path)

	def save_predictions_det(self):

		# save predictions in COCO format
		img_path2idx = {v:k for k, v in self._data_df['Path'].to_dict().items()}

		coco_results = []
		coco_result = {'image_id': None, 'category_id': None, 'bbox': None, 'score': None}

		for img_path, result in self.results.items():
			if not img_path in img_path2idx:
				continue
			for item in result:
				classes = item.boxes.cls.cpu().tolist()
				confs = item.boxes.conf.cpu().tolist()
				boxes = item.boxes.xyxy.cpu().tolist()
				for cl, conf, box in zip(classes, confs, boxes):
					coco_result['image_id'] = img_path2idx[img_path]
					coco_result['category_id'] = int(cl)
					coco_result['bbox'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]] #convert to left-corner-width-height
					coco_result['score'] = conf
					coco_results.append(coco_result.copy())
		with open(self._results_path, 'w') as fid:
			json.dump(coco_results, fid, ensure_ascii=False, indent=2)

	def save_predictions_seg(self):
		# save predictions in COCO format
		img_path2idx = {v:k for k, v in self._data_df['Path'].to_dict().items()}
		coco_results = []
		coco_result = {'image_id': None, 'category_id': None, 'seg': None, 'score': None}
		for img_path, result in self.results.items():
			if not img_path in img_path2idx:
				continue
			for item in result:
				segments = []
				try:
					segments = item.masks.xy.cpu().tolist()
				except:
					pass

# run from scripts/models/ultralytics_models
# object detection: yolov3.pt yolov5x.pt yolov5x6u.pt yolov8x.pt, yolov10x.pt, yolov11x.pt rtdetr-x.pt
# segmentation: sam_l sam2_l mobile_sam yolov8x-seg
# classification: yolov8x-cls, yolo11x-cls

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Run ultralytics algorithms on an image or a list of images')
	parser.add_argument('--model_name', help='Model to run')
	parser.add_argument('--task', help='Task: classify, detect, segment')
	parser.add_argument('--data_version', default='v5', help='Data version: default v3')
	parser.add_argument('--input_path', help='Path to excel file with input')
	parser.add_argument('--results_path', help='Path to excel file with results')

	args = vars(parser.parse_args())
	print(args)

	#model_name = 'yolov5x'
	m = UltralyticsModel(**args)
	m.predict()
	m.save_results()
	#m.save_predictions_COCO()


	#im_class = ImageClassification(model_name=args.model_name)
	#if args.img_path:
	#	im_class.predict_img(img_path=args.img_path)
	#else:
	#	im_class.predict_list(input_path=args.input_path, results_path=args.results_path)