import argparse
import os.path as osp
import numpy as np

from pygda.datasets import CitationDataset

from pygda.models import PairAlign
from pygda.metrics import eval_micro_f1, eval_macro_f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

parser = argparse.ArgumentParser()

# model agnostic params
parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--nhid', type=int, default=256, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--source', type=str, default='tnbc6k', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='tnbc1', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs')
parser.add_argument('--filename', type=str, default='test.txt', help='store results into file')

# model specific params
parser.add_argument('--cls_dim', type=int, default=128, help='hidden dimension for classification layer')
parser.add_argument('--cls_layers', type=int, default=2, help='total number of cls layers in model')
parser.add_argument('--ew_start', type=int, default=0, help='starting epoch for edge reweighting')
parser.add_argument('--ew_freq', type=int, default=10, help='frequency for edge reweighting')
parser.add_argument('--lw_start', type=int, default=0, help='starting epoch for label reweighting')
parser.add_argument('--lw_freq', type=int, default=10, help='frequency for label reweighting')
parser.add_argument('--pooling', type=str, default='mean', help='aggregation in gnn')
parser.add_argument('--ew_type', type=str, default='pseudobeta', help='use the true edge weight or not')
parser.add_argument('--rw_lmda', type=float, default=1.0, help='trade-off parameter for edge reweight')
parser.add_argument('--ls_lambda', type=float, default=1.0, help='regularize the distance to 1 in w optimization')
parser.add_argument('--lw_lambda', type=float, default=0.005, help='regularize the distance to 1 in beta optimization')
parser.add_argument('--label_rw', type=bool, default=True, help='reweight the label or not')
parser.add_argument('--edge_rw', type=bool, default=True, help='reweight the edge in source graph or not')
parser.add_argument('--gamma_reg', type=float, default=0.0001, help='mimic the variance of the edges to normalize the weight')
parser.add_argument('--weight_CE_src', type=bool, default=True, help='reweight the loss by src class or not')
parser.add_argument('--backbone', type=str, default='GS', help='the backbone of PairAlign')
args = parser.parse_args()

# load data
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.source)
source_dataset = CitationDataset(path, args.source)
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.target)
target_dataset = CitationDataset(path, args.target)
source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)

num_features = source_data.x.size(1)
num_classes = len(np.unique(source_data.y.cpu().numpy()))

model = PairAlign(
    in_dim=num_features,
    hid_dim=args.nhid,
    num_classes=num_classes,
    num_layers=args.num_layers,
    weight_decay=args.weight_decay,
    lr=args.lr,
    dropout=args.dropout_ratio,
    epoch=args.epochs,
    device=args.device,
    cls_dim=args.cls_dim,
    cls_layers=args.cls_layers,
    ew_start=args.ew_start,
    ew_freq=args.ew_freq,
    lw_start=args.lw_start,
    lw_freq=args.lw_freq,
    pooling=args.pooling,
    ew_type=args.ew_type,
    rw_lmda=args.rw_lmda,
    ls_lambda=args.ls_lambda,
    lw_lambda=args.lw_lambda,
    label_rw=args.label_rw,
    edge_rw=args.edge_rw,
    gamma_reg=args.gamma_reg,
    weight_CE_src=args.weight_CE_src,
    backbone=args.backbone
    )
global_global_accuracy = 0
global_precision = 0
global_recall = 0
global_f1 = 0
global_mi_f1 = 0
global_ma_f1 = 0
global_best_accuracy = 0
global_conf_matrix = []

model.fit(source_data, target_data)
# evaluate the performance
logits, labels = model.predict(target_data)
preds = logits.argmax(dim=1)
mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)
labels = labels.cpu().numpy()
preds = preds.cpu().numpy()
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average='macro')
recall = recall_score(labels, preds, average='macro')
f1 = f1_score(labels, preds, average='macro')
conf_matrix = confusion_matrix(labels, preds)

