import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import json
import copy

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.AdamW import AdamW
from lib.relational_transformer import Relational_Transformer
from lib.mlp import MLP
from lib.gnn import GraphEDWrapper, RBP, BiGED
from lib.utils import update_train_config
from lib.metrics import mean_recall_at_k, n_recall_at_k
from sklearn.metrics import average_precision_score

"""------------------------------------some settings----------------------------------------"""
conf = Config()
conf = update_train_config(conf)

writer = SummaryWriter(log_dir=os.path.join('runs', conf.save_path))

AG_dataset_train = AG(mode="train", data_path=conf.data_path, filter_nonperson_box_frame=True,
                      num_frames=conf.num_frames, infer_last=conf.infer_last)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=conf.num_workers,
                                               collate_fn=cuda_collate_fn, pin_memory=False)

AG_dataset_test = AG(mode="test", data_path=conf.data_path, filter_nonperson_box_frame=True,
                     num_frames=conf.num_frames, infer_last=conf.infer_last)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=conf.num_workers,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone 
object_detector = detector(object_classes=AG_dataset_train.object_classes).to(device=gpu_device)
object_detector.eval()

if conf.model_type == 'transformer':
    model = Relational_Transformer(obj_classes=AG_dataset_train.object_classes,
                   enc_layer_num=conf.enc_layer,
                   dec_layer_num=conf.dec_layer,
                   semantic=conf.semantic,
                   concept_net=conf.concept_net,
                   cross_attention=conf.cross_attention).to(device=gpu_device)
elif conf.model_type == 'mlp':
    model = MLP(conf.mlp_layers, obj_classes=AG_dataset_train.object_classes, semantic=conf.semantic, concept_net=conf.concept_net).to(device=gpu_device)
elif conf.model_type == 'GNNED':
    model = GraphEDWrapper(obj_classes=AG_dataset_train.object_classes).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'RBP':
    model = RBP(obj_classes=AG_dataset_train.object_classes).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'BiGED':
    model = BiGED(obj_classes=AG_dataset_train.object_classes, out_size = conf.emb_out ).to(device=gpu_device) # default is using semantic

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
for epoch in range(conf.nepoch):
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

        entry['pool_type'] = conf.pool_type
        pred = model(entry)

        action_class_distribution = pred["action_class_distribution"]
        gt_action_class_label = -torch.ones([len(gt_annotation), 157], dtype=torch.long).to(device=action_class_distribution.device)

        for i in range(len(gt_annotation)):
            gt_action_class_label[i, : len(gt_annotation[i][-1]['action_class'])] = torch.tensor(gt_annotation[i][-1]['action_class'])


        loss = mlm_loss(action_class_distribution, gt_action_class_label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        writer.add_scalar('loss/train', loss.item(), iteration)
        iteration += 1

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))

    model.eval()

    object_detector.is_train = False
    all_frame_action_class_distribution = None
    all_test_frame_gt_action_class = None

    preds = None
    gts = None

    gt_action_set = []
    recall = []

    gt_stats = np.zeros(157)
    pred_stats = np.zeros(157)

    with torch.no_grad():
        for b, data in enumerate(dataloader_test):

            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            entry['pool_type'] = conf.pool_type

            pred = model(entry)

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

    model_result_path = os.path.join(conf.save_path, 'model_{}_results'.format(epoch))
    if not os.path.exists(model_result_path):
            os.makedirs(model_result_path)

    with open(os.path.join(model_result_path, 'model_{}.txt'.format(epoch)), 'w') as f: 
        f.write("Individual class scores: {}\n".format(individual_class_scores))
        f.write("Overall mean average precision across classes: {}\n".format(mean_ap))
        f.write("Overall recall @ K={}: {}\n".format(conf.top_k, overall_recall))
        f.write("Overall mean recall @ K={}: {}\n".format(conf.top_k, mean_recall))
        f.write("Individual overall recall for N action: {}".format(n_overall_recall))

    if conf.lr_scheduler:
        scheduler.step(mean_ap)



