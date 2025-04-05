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
from lib.utils import update_train_config
from lib.AdamW import AdamW
from lib.relational_transformer import Relational_Transformer_Veri
from lib.mlp import MLP_Veri
from lib.gnn import BiGED_Veri, RBP_Veri, GNNED_Veri
from sklearn.metrics import average_precision_score

import clip
from PIL import Image


"""------------------------------------some settings----------------------------------------"""
conf = Config()
conf = update_train_config(conf)

writer = SummaryWriter(log_dir=os.path.join('runs', conf.save_path))

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
                                        semantic=conf.semantic,
                                        concept_net=conf.concept_net).to(device=gpu_device)
elif conf.model_type == 'mlp':
    model = MLP_Veri(conf.mlp_layers, obj_classes=AG_dataset_train.object_classes, semantic=conf.semantic).to(device=gpu_device)
elif conf.model_type == 'BiGED':
    model = BiGED_Veri(obj_classes=AG_dataset_train.object_classes, out_size = conf.emb_out).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'RBP':
    model = RBP_Veri(obj_classes=AG_dataset_train.object_classes).to(device=gpu_device)
elif conf.model_type == 'GNNED':
    model = GNNED_Veri(obj_classes=AG_dataset_train.object_classes).to(device=gpu_device) # default is using semantic

mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)


if conf.lr_scheduler:
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

iteration = 0
destination_str = os.path.split(conf.save_path)[-1]

action_names_list = open('Charades_annotations/Charades_v1_classes.txt').readlines()  
action_names = []
for action_name in action_names_list:
    action_names.append(action_name[5:-1])

with torch.no_grad():
    text = clip.tokenize(action_names).to(gpu_device)
    text_features = clip_model.encode_text(text)

for epoch in range(conf.nepoch):
    total_loss = 0.
    ep_start_time = datetime.now().strftime("%H:%M:%S")
    model.train()
    object_detector.is_train = True
    start = time.time()

    for b, data in enumerate(dataloader_train):    

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]   
        
        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        entry['pool_type'] = conf.pool_type
        pred = model(entry, class_enc=text_features)

        action_class_distribution = pred["action_class_distribution"]
        gt_action_class_label = -torch.ones([len(gt_annotation), 157], dtype=torch.long).to(device=action_class_distribution.device)

        for i in range(len(gt_annotation)):
            gt_action_class_label[i, : len(gt_annotation[i][-1]['action_class'])] = torch.tensor(gt_annotation[i][-1]['action_class'])


        loss = mlm_loss(action_class_distribution, gt_action_class_label)
        
        if conf.model_type == 'mlp_jvs' or conf.model_type == 'transformer_jvs':
            loss = loss + conf.jvs_lr * pred["jvs_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        writer.add_scalar('loss/train', loss.item(), iteration)
        iteration += 1.        
        total_loss += loss.item()

        dev = 10
        if b % dev == 0 and b >= dev:
            time_per_batch = (time.time() - start) / b
            print("\n[{}] [{}] e{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch [{}] [{:.1f}] [{:.1f}]".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S") ,
                ep_start_time,
                epoch, 
                b, 
                len(dataloader_train),
                time_per_batch, 
                len(dataloader_train) * time_per_batch / 60,
                destination_str, loss.item(), total_loss/iteration)
                ,flush=True)

    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch + conf.resume_epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch + conf.resume_epoch))    

    model.eval()

    object_detector.is_train = False
    all_frame_action_class_distribution = None
    all_test_frame_gt_action_class = None

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

            action_class_distribution = pred["action_class_distribution"].to('cpu').numpy()

            test_action_class_labels = torch.zeros(len(gt_annotation), 157)
            for i in range(len(gt_annotation)):
                test_action_class_labels[i].index_fill_(0, torch.tensor(gt_annotation[i][-1]['action_class']).to('cpu'), 1)
            test_action_class_labels = test_action_class_labels.numpy()

            if all_frame_action_class_distribution is None and all_test_frame_gt_action_class is None:
                all_frame_action_class_distribution = action_class_distribution
                all_test_frame_gt_action_class = test_action_class_labels
            else:
                all_frame_action_class_distribution = np.concatenate((all_frame_action_class_distribution, action_class_distribution), axis=0)
                all_test_frame_gt_action_class = np.concatenate((all_test_frame_gt_action_class, test_action_class_labels), axis=0)

        individual_class_scores = average_precision_score(all_test_frame_gt_action_class, all_frame_action_class_distribution, average=None)
        average_precision = np.mean(individual_class_scores)

    model_result_path = os.path.join(conf.save_path, 'model_{}_results'.format(epoch + conf.resume_epoch))
    if not os.path.exists(model_result_path):
            os.makedirs(model_result_path)

    with open(os.path.join(model_result_path, 'model_{}.txt'.format(epoch + conf.resume_epoch)), 'w') as f: 
        f.write("Individual class scores: {}\n".format(individual_class_scores))
        f.write("Average precision across classes: {}".format(average_precision))

    if conf.lr_scheduler:
        print("step scheduler")
        scheduler.step(average_precision)


