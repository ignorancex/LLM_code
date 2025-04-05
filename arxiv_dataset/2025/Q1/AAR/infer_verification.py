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
from lib.AdamW import AdamW
from lib.relational_transformer import Relational_Transformer_Veri
from lib.mlp import MLP_Veri
from lib.gnn import BiGED_Veri, RBP_Veri, GNNED_Veri
from lib.utils import get_visualization_and_results, count_parameters, update_test_config
from lib.metrics import mean_recall_at_k
from sklearn.metrics import average_precision_score

import clip
from PIL import Image

"""------------------------------------some settings----------------------------------------"""
conf = Config()
conf = update_test_config(conf)

gt_stats = np.zeros(157)
pred_stats = np.zeros(157)

writer = SummaryWriter(log_dir=os.path.join('runs-test', conf.save_path))

AG_dataset_train = AG(mode="train", data_path=conf.data_path, filter_nonperson_box_frame=True,
                      num_frames=conf.num_frames, infer_last=conf.infer_last, task=conf.task)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=conf.num_workers,
                                               collate_fn=cuda_collate_fn, pin_memory=False)

AG_dataset_test = AG(mode="test", data_path=conf.data_path, filter_nonperson_box_frame=True,
                     num_frames=conf.num_frames, infer_last=conf.infer_last, task=conf.task)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=conf.num_workers,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone # train parameter does not affect predcls
object_detector = detector(object_classes=AG_dataset_train.object_classes).to(device=gpu_device)
object_detector.eval()

clip_model, clip_preprocess = clip.load("ViT-B/32", device=gpu_device)

if conf.model_type == 'transformer':
    model = Relational_Transformer_Veri(obj_classes=AG_dataset_train.object_classes,
                                        enc_layer_num=conf.enc_layer,
                                        dec_layer_num=conf.dec_layer,
                                        semantic=conf.semantic).to(device=gpu_device)
elif conf.model_type == 'mlp':
    model = MLP_Veri(conf.mlp_layers, obj_classes=AG_dataset_train.object_classes, semantic=conf.semantic).to(device=gpu_device)
elif conf.model_type == 'BiGED':
    model = BiGED_Veri(obj_classes=AG_dataset_train.object_classes, out_size = conf.emb_out).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'RBP':
    model = RBP_Veri(obj_classes=AG_dataset_train.object_classes).to(device=gpu_device)
elif conf.model_type == 'GNNED':
    model = GNNED_Veri(obj_classes=AG_dataset_train.object_classes).to(device=gpu_device) # default is using semantic


action_names_list = open('Charades_annotations/Charades_v1_classes.txt').readlines()  
action_names = []
for action_name in action_names_list:
    action_names.append(action_name[5:-1])

with torch.no_grad():
    text = clip.tokenize(action_names).to(gpu_device)
    text_features = clip_model.encode_text(text)

if conf.model_path is not None:
    print('resume from ... {}'.format(conf.model_path))
    ckpt = torch.load(os.path.join(conf.conf_path, conf.model_path), map_location=gpu_device) 
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print('resume from ... {} done'.format(conf.model_path))  

model.eval()

object_detector.is_train = False
preds = None
gts = None
recall = []

with torch.no_grad():
    for b, data in tqdm(enumerate(dataloader_test)):

            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            entry['pool_type'] = conf.pool_type

            pred = model(entry, class_enc=text_features)

            action_class_distribution = torch.nan_to_num(pred["action_class_distribution"]).to('cpu').numpy()

            gt_labels = torch.zeros(len(gt_annotation), 157)
            for i in range(len(gt_annotation)):
                gt_labels[i].index_fill_(0, torch.tensor(gt_annotation[i][-1]['action_class']).to('cpu'), 1)
            gt_labels = gt_labels.numpy()
    
            recall = recall + mean_recall_at_k(action_class_distribution, gt_labels, conf.top_k, gt_stats, pred_stats)             
            
            if preds is None and gts is None:
                preds = action_class_distribution
                gts = gt_labels
            else:
                preds = np.concatenate((preds, action_class_distribution), axis=0)
                gts = np.concatenate((gts, gt_labels), axis=0)

# calculation of map
individual_class_scores = average_precision_score(gts, preds, average=None)
average_precision = np.mean(individual_class_scores)

# calculation of recall.
recall_all = np.array(recall).mean()

# calculation of mean recall
mean_recall = []
for ci in range(157):                
    if gt_stats[ci] > 0.:
        class_recall = pred_stats[ci] / gt_stats[ci]
        mean_recall.append(class_recall)
mean_recall = np.array(mean_recall).mean()

model_result_path = os.path.join(conf.save_path, 'model_{}_results'.format(conf.resume_epoch))

if not os.path.exists(model_result_path):
    os.makedirs(model_result_path)

with open(os.path.join(model_result_path, 'new_model_{}.txt'.format(conf.resume_epoch)), 'w') as f: 
    f.write("Individual class scores: {}\n".format(individual_class_scores))
    f.write("Average precision across classes: {}\n".format(average_precision))
    f.write("recall_@{}: {}\n".format(conf.top_k, recall_all))
    f.write("mean_recall_@{}: {}".format(conf.top_k, mean_recall))

with open(os.path.join(model_result_path, 'recall_at{}_model_{}.txt'.format(conf.top_k, conf.resume_epoch)), 'w') as f:     
    f.write("{}".format(recall_all))
with open(os.path.join(model_result_path, 'mean_recall_at{}_model_{}.txt'.format(conf.top_k, conf.resume_epoch)), 'w') as f:     
    f.write("{}".format(mean_recall))

    

    

