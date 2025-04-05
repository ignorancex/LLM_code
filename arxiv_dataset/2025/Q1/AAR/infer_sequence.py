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

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.AdamW import AdamW
from lib.relational_transformer import Relational_Transformer
from lib.mlp import MLP
from lib.gnn import GraphEDWrapper, RBP, BiGED
from lib.seq_models import GRUDecoder, SeqTransformerDecoder
from lib.utils import get_visualization_and_results, count_parameters, update_test_config
from sklearn.metrics import average_precision_score

"""------------------------------------some settings----------------------------------------"""
conf = Config()
conf = update_test_config(conf)

AG_dataset = AG(mode="test", data_path=conf.data_path, filter_nonperson_box_frame=True,
                num_frames=conf.num_frames, infer_last=conf.infer_last, task=conf.task)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')

object_detector = detector(object_classes=AG_dataset.object_classes).to(device=gpu_device)
object_detector.eval()

if conf.model_type == 'transformer':
    model = AAT(obj_classes=AG_dataset.object_classes,
                   enc_layer_num=conf.enc_layer,
                   dec_layer_num=conf.dec_layer,
                   semantic=conf.semantic,
                   concept_net=conf.concept_net).to(device=gpu_device)
elif conf.model_type == 'mlp':
    model = MLP(conf.mlp_layers, obj_classes=AG_dataset.object_classes, semantic=conf.semantic, concept_net=conf.concept_net).to(device=gpu_device)
elif conf.model_type == 'GNNED' or conf.model_type == 'GraphEDWrapper':
    model = GraphEDWrapper(obj_classes=AG_dataset.object_classes).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'RBP' or conf.model_type == 'BiLinearBase':
    model = RBP(obj_classes=AG_dataset.object_classes).to(device=gpu_device) # default is using semantic
elif conf.model_type == 'BiGED':
    model = BiGED(obj_classes=AG_dataset.object_classes, out_size = conf.emb_out ).to(device=gpu_device) # default is using semantic
    conf.hidden_dim = 1536

# load previous model parameters and use it for inference only
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

print('x'*30 + ' loading sequence checkpoint ' + 'x'*30)
seq_ckpt = torch.load(os.path.join(conf.conf_path, conf.seq_model_path), map_location=gpu_device)
seq_model.load_state_dict(seq_ckpt['state_dict'])
print('x'*30 + ' loaded sequence checkpoint ' + 'x'*30)
seq_model.eval()

overall_test_accuracy = 0.0

with torch.no_grad():
    for b, data in tqdm(enumerate(dataloader)):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]

        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        entry['pool_type'] = conf.pool_type

        pred = model(entry)

        action_class_distribution = pred["action_class_distribution"] # actions
        relational_feats = entry["relational_feats"] # hidden state

        gt_action_class_label = []
        for i in range(len(gt_annotation)):
            gt_action_class_label.append(torch.tensor(gt_annotation[i][-1]['action_class']))
        gt_action_class_label = torch.nn.utils.rnn.pad_sequence(gt_action_class_label, padding_value=-1).to(gpu_device) # correct

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

with open(os.path.join(conf.save_path, conf.seq_model_path.split('.')[0] + '.txt'), 'w') as f: 
    f.write("Test accuracy: {:.2f}%\n".format((overall_test_accuracy/len(dataloader))*100))


