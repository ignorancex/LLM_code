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
import random

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.AdamW import AdamW
from lib.relational_transformer import Relational_Transformer
from lib.mlp import MLP
from lib.gnn import GraphEDWrapper, RBP, BiGED
from lib.utils import update_train_config, count_parameters
from lib.seq_models import GRUDecoder, SeqTransformerDecoder
from sklearn.metrics import average_precision_score

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
    conf.hidden_dim = 1536

# load previous model parameters and use it for inference only
if conf.task == 'sequence':
    ## if conf.model_path is not None:    
    print('x'*30 + ' loading checkpoint ' + 'x'*30)
    # os.path.join(conf.conf_path, conf.model_path)
    ckpt = torch.load(conf.model_path, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print('x'*30 + ' loaded checkpoint ' + 'x'*30)
    model.eval()

    if conf.seq_model == 'gru':
        seq_model = GRUDecoder(input_size=158, hidden_size=conf.hidden_dim, num_layers=conf.seq_layer, num_classes=158) # instead of 157 158 because 0 - 157 classes 157 is stop
    else:
        seq_model = SeqTransformerDecoder(input_size=conf.hidden_dim, hidden_size=conf.hidden_dim, num_layers=conf.seq_layer, 
                                          num_mlp_layers=conf.seq_model_mlp_layers, num_heads=8, num_classes=158, dropout=0.1)

    seq_model.to(gpu_device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

# # optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(seq_model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(seq_model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(seq_model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

if conf.lr_scheduler:
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

iteration = 0
for epoch in range(conf.nepoch):
    seq_model.train()

    start = time.time()
    overall_train_accuracy = 0.0
    overall_test_accuracy = 0.0

    for b, data in enumerate(dataloader_train):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]

        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

            entry['pool_type'] = conf.pool_type
            pred = model(entry) # set model to eval and freezed

        gt_action_class_label = []
        for i in range(len(gt_annotation)):
            gt_action_class_label.append(torch.tensor(gt_annotation[i][-1]['action_class']))
        gt_action_class_label = torch.nn.utils.rnn.pad_sequence(gt_action_class_label, padding_value=-1).to(gpu_device)

        # get max length of sequence within the video
        max_len = gt_action_class_label.size(0)

        outputs = []

        if conf.seq_model == 'gru':

            action_class_distribution = pred["action_class_distribution"] # actions
            relational_feats = entry["relational_feats"] # hidden state for gru / query for transformer

            out = torch.zeros(action_class_distribution.size(0), 158)
            out[:, :action_class_distribution.size(1)] = action_class_distribution
            out = out.unsqueeze(0).to(gpu_device)

            hidden = seq_model.init_hidden(relational_feats)

            for i in range(max_len):      
                out, hidden = seq_model(out, hidden)
                outputs.append(out)

        else:
            for i in range(max_len):
                entry["relational_feats"], scores = seq_model(entry, conf)
                outputs.append(scores)

        output = (torch.stack(outputs, dim=0).squeeze()).view(-1, 158)
        loss = criterion(output, gt_action_class_label.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq_model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        writer.add_scalar('loss/train', loss.item(), iteration)

        _, predicted = torch.max(output.detach(), 1)
        labels = gt_action_class_label.view(-1).detach()
        mask = (labels != -1)
        accuracy = (predicted[mask] == labels[mask]).float().mean()
        overall_train_accuracy += accuracy
        writer.add_scalar('acc/train', accuracy.item(), iteration)

        iteration += 1

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

    torch.save({"state_dict": seq_model.state_dict()}, os.path.join(conf.save_path, "seq_model_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))
    print("training accuracy after {} epochs is : {:.2f}%".format(epoch, (overall_train_accuracy/len(dataloader_train))*100))

    # # set to evaluation
    seq_model.eval()

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

            gt_action_class_label = []
            for i in range(len(gt_annotation)):
                gt_action_class_label.append(torch.tensor(gt_annotation[i][-1]['action_class']))
            gt_action_class_label = torch.nn.utils.rnn.pad_sequence(gt_action_class_label, padding_value=-1).to(gpu_device)

            # get max length of sequence within the video
            max_len = gt_action_class_label.size(0)

            outputs = []

            if conf.seq_model == 'gru':
                action_class_distribution = pred["action_class_distribution"] # actions
                relational_feats = entry["relational_feats"] # hidden state

                out = torch.zeros(action_class_distribution.size(0), 158)
                out[:, :action_class_distribution.size(1)] = action_class_distribution
                out = out.unsqueeze(0).to(gpu_device)

                hidden = seq_model.init_hidden(relational_feats)

                for i in range(max_len):
                    out, hidden = seq_model(out, hidden)
                    outputs.append(out) # because it does it in batches thats why need to do all together.
            else:
                for i in range(max_len):
                    entry["relational_feats"], scores = seq_model(entry, conf)
                    outputs.append(scores)

            output = (torch.stack(outputs, dim=0).squeeze()).view(-1, 158)

            _, predicted = torch.max(output.detach(), 1)
            labels = gt_action_class_label.view(-1).detach()
            ignore = torch.tensor([-1, 157]).to(gpu_device)
            mask = ~torch.isin(labels, ignore)

            accuracy = (predicted[mask] == labels[mask]).float().mean()
            overall_test_accuracy += accuracy

    model_result_path = os.path.join(conf.save_path, 'model_{}_results'.format(epoch))
    if not os.path.exists(model_result_path):
        os.makedirs(model_result_path)

    with open(os.path.join(model_result_path, 'model_{}.txt'.format(epoch)), 'w') as f: 
        f.write("Test accuracy: {:.2f}%\n".format((overall_test_accuracy/len(dataloader_test))*100))

    if conf.lr_scheduler:
        scheduler.step((overall_test_accuracy/len(dataloader_test)))


