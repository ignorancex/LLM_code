import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.AdamW import AdamW
from lib.relational_transformer import Relational_Transformer
from lib.mlp import MLP
from lib.gnn import GraphEDWrapper, RBP, BiGED
from lib.utils import get_visualization_and_results, count_parameters, update_test_config
from lib.metrics import mean_recall_at_k, n_recall_at_k
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore') 


"""------------------------------------some settings----------------------------------------"""
conf = Config()
conf = update_test_config(conf)

if not os.path.exists(conf.save_path):
    os.makedirs(conf.save_path)

AG_dataset = AG(mode="test", data_path=conf.data_path, filter_nonperson_box_frame=True,
                num_frames=conf.num_frames, infer_last=conf.infer_last)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')

object_detector = detector(object_classes=AG_dataset.object_classes).to(device=gpu_device)
object_detector.eval()

if conf.model_type == 'transformer':
    model = Relational_Transformer(obj_classes=AG_dataset.object_classes,
                   enc_layer_num=conf.enc_layer,
                   dec_layer_num=conf.dec_layer,
                   semantic=conf.semantic,
                   concept_net=conf.concept_net,
                   cross_attention=conf.cross_attention).to(device=gpu_device)
elif conf.model_type == 'mlp':
    model = MLP(conf.mlp_layers, obj_classes=AG_dataset.object_classes, semantic=conf.semantic, concept_net=conf.concept_net).to(device=gpu_device)
elif conf.model_type == 'GNNED' or conf.model_type == 'GraphEDWrapper':
    model = GraphEDWrapper(obj_classes=AG_dataset.object_classes).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'RBP' or conf.model_type == 'BiLinearBase':
    model = RBP(obj_classes=AG_dataset.object_classes).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'BiGED':
    model = BiGED(obj_classes=AG_dataset.object_classes, out_size = conf.emb_out).to(device=gpu_device) # default is using semantic
    
model.eval()

if conf.model_path is not None:
    print('x'*30 + ' loading checkpoint ' + 'x'*30)
    ckpt = torch.load(os.path.join(conf.conf_path, conf.model_path), map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print('x'*30 + ' loaded checkpoint ' + 'x'*30)

preds = None
gts = None

gt_action_set = []
recall = []

gt_stats = np.zeros(157)
pred_stats = np.zeros(157)

with torch.no_grad():
    for b, data in enumerate(tqdm(dataloader)):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]

        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None, predict_flag=conf.predict_flag)
        entry['pool_type'] = conf.pool_type

        pred = model(entry)

        if conf.visualize:
            get_visualization_and_results(conf.data_path, conf.save_path, gt_annotation, AG_dataset, pred["action_class_distribution"])

        action_class_distribution = pred["action_class_distribution"].to('cpu').numpy()


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

    # compute mean average precision score based on the individual classes
    individual_class_scores = average_precision_score(gts, preds, average=None) # average='macro' same as np.mean
    mean_ap = np.mean(individual_class_scores)

    # compute recall for all samples
    overall_recall = np.array(recall).mean()

    mean_recall = []
    for ci in range(157):                
        if gt_stats[ci] > 0.:
            class_recall = pred_stats[ci] / gt_stats[ci]
            mean_recall.append(class_recall)
    mean_recall = np.array(mean_recall).mean()

    # compute overall recall for individual group of actions from 1 to N with top_k
    n_overall_recall = n_recall_at_k(preds, gts, conf.top_k, gt_action_set)

    with open(os.path.join(conf.save_path, conf.model_path.split('.')[0] + '.txt'), 'w') as f: 
        f.write("Individual class scores: {}\n".format(individual_class_scores))
        f.write("Overall mean average precision across classes: {}\n".format(mean_ap))
        f.write("Overall recall @ K={}: {}\n".format(conf.top_k, overall_recall))
        f.write("Overall mean recall @ K={}: {}\n".format(conf.top_k, mean_recall))
        f.write("Individual overall recall for N action: {}".format(n_overall_recall))