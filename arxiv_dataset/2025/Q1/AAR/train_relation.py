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
from lib.utils import update_train_config
from sklearn.metrics import average_precision_score

"""------------------------------------some settings----------------------------------------"""
conf = Config()
conf = update_train_config(conf)

writer = SummaryWriter(log_dir=os.path.join('runs', conf.save_path))

AG_dataset_train = AG(mode="train", data_path=conf.data_path, filter_nonperson_box_frame=True,
                      num_frames=conf.num_frames)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=conf.num_workers,
                                               collate_fn=cuda_collate_fn, pin_memory=False)

AG_dataset_test = AG(mode="test", data_path=conf.data_path, filter_nonperson_box_frame=True,
                     num_frames=conf.num_frames)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=conf.num_workers,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone
object_detector = detector(object_classes=AG_dataset_train.object_classes).to(device=gpu_device)
object_detector.eval()

model = Relational(obj_classes=AG_dataset_train.object_classes)

iteration = 0

destination_str = os.path.split(conf.save_path)[-1]
for epoch in range(1):
    ep_start_time = datetime.now().strftime("%H:%M:%S")
    model.train()

    start = time.time()

    for b, data in enumerate(dataloader_train):    

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]
        
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        gt_action_class_label = torch.zeros([len(gt_annotation), 157], dtype=torch.long)
        for i in range(len(gt_annotation)):
            gt_action_class_label[i, gt_annotation[i][-1]['action_class']] = 1

        entry['gt_action_class_label'] = gt_action_class_label
        
        pred = model(entry)   
        action_class_distribution = pred["action_class_distribution"]
        iteration += 1        

        dev = 100
        if b % dev == 0 and b >= dev:
            time_per_batch = (time.time() - start) / b
            print("\n[{}] [{}] e{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch [{}]".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S") ,
                ep_start_time,
                epoch, 
                b, 
                len(dataloader_train),
                time_per_batch, 
                len(dataloader_train) * time_per_batch / 60,
                destination_str)
                ,flush=True)

    torch.save({"model": model}, os.path.join(conf.save_path, "model_final.tar"))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))    

    model.eval()

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

            pred = model(entry)

            action_class_distribution = pred["action_class_distribution"].numpy()

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

    model_result_path = os.path.join(conf.save_path, 'model_{}_results'.format(epoch))
    if not os.path.exists(model_result_path):
            os.makedirs(model_result_path)

    with open(os.path.join(model_result_path, 'model_{}.txt'.format(epoch)), 'w') as f: 
        f.write("Individual class scores: {}\n".format(individual_class_scores))
        f.write("Average precision across classes: {}".format(average_precision))

   
