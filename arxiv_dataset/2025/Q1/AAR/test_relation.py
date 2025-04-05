import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import json
import copy
from tqdm import tqdm
from datetime import datetime

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.gnn import Relational
from lib.utils import get_visualization_and_results, update_test_config
from lib.metrics import mean_recall_at_k, n_recall_at_k
from sklearn.metrics import average_precision_score


import warnings
warnings.filterwarnings('ignore') 

conf = Config()
conf = update_test_config(conf)

writer = SummaryWriter(log_dir=os.path.join('runs', conf.save_path))

AG_dataset = AG(mode="test", data_path=conf.data_path, filter_nonperson_box_frame=True,
				num_frames=conf.num_frames, infer_last=conf.infer_last)
dataloader_test = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=conf.num_workers,
											  collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone # train parameter does not affect predcls
object_detector = detector(object_classes=AG_dataset.object_classes).to(device=gpu_device)
object_detector.eval()

model = Relational(obj_classes=AG_dataset.object_classes)
if conf.model_path is not None:
	print('x'*30 + ' loading checkpoint ' + 'x'*30)
	model = torch.load(os.path.join(conf.conf_path, conf.model_path))['model']
	print('x'*30 + ' loaded checkpoint ' + 'x'*30)

model.eval()

preds = None
gts = None

gt_action_set = []
recall = []

gt_stats = np.zeros(157)
pred_stats = np.zeros(157)


with torch.no_grad():
	for b, data in tqdm(enumerate(dataloader_test)):

		im_data = copy.deepcopy(data[0].cuda(0))
		im_info = copy.deepcopy(data[1].cuda(0))
		gt_boxes = copy.deepcopy(data[2].cuda(0))
		num_boxes = copy.deepcopy(data[3].cuda(0))
		gt_annotation = AG_dataset.gt_annotations[data[4]]

		entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)           

		pred = model(entry)

		action_class_distribution = pred["action_class_distribution"].numpy()

		if conf.visualize:
			get_visualization_and_results(conf.data_path, conf.save_path, gt_annotation, AG_dataset, pred["action_class_distribution"])

		gt_labels = torch.zeros(len(gt_annotation), 157)
		for i in range(len(gt_annotation)):
			gt_action_set.append(gt_annotation[i][-1]['action_class'])
			gt_labels[i].index_fill_(0, torch.tensor(gt_annotation[i][-1]['action_class']).to('cpu'), 1)
		gt_labels = gt_labels.numpy()

		recall = recall + mean_recall_at_k(action_class_distribution, gt_labels, conf.top_k, gt_stats, pred_stats)

		if preds is None and gts is None:
			preds = action_class_distribution
			gts = gt_labels
		else:
			preds = np.concatenate((preds, action_class_distribution), axis=0)
			gts = np.concatenate((gts, gt_labels), axis=0)

	individual_class_scores = average_precision_score(gts, preds, average=None)
	mean_ap = np.mean(individual_class_scores)

	# compute recall for all samples
	overall_recall = np.array(recall).mean()

	mean_recall = []
	for ci in range(157):                
		if gt_stats[ci] > 0.:
			class_recall = pred_stats[ci] / gt_stats[ci]
			mean_recall.append(class_recall)
	mean_recall = np.array(mean_recall).mean()

	# compute overall recall for individual group of actions from 1 to N
	n_overall_recall = n_recall_at_k(preds, gts, conf.top_k, gt_action_set)

with open(os.path.join(conf.save_path, conf.model_path.split('.')[0] + '.txt'), 'w') as f: 
	f.write("Individual class scores: {}\n".format(individual_class_scores))
	f.write("Overall mean average precision across classes: {}\n".format(mean_ap))
	f.write("Overall recall @ K={}: {}\n".format(conf.top_k, overall_recall))
	f.write("Overall mean recall @ K={}: {}\n".format(conf.top_k, mean_recall))
	f.write("Individual overall recall for N action: {}".format(n_overall_recall))

   
 