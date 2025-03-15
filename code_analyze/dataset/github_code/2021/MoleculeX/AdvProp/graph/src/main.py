import argparse
import torch
import torch.nn.functional as F
from texttable import Texttable
import sys

from datasets import *
from train_eval import run_classification, run_regression
from torch_scatter import scatter_mean
from model import *




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="qm8", help='dataset name')

### If split ready is True, use (1); Otherwise, use (2).###
parser.add_argument('--split_ready', action='store_true', default=False, help='specify it to be true if you provide three files for train/val/test')
####################################################################

### (1) The following arguments are used when split_ready==True.###
parser.add_argument('--trainfile', type=str, help='path to the preprocessed training file (Pytorch Geometric Data)')
parser.add_argument('--validfile', type=str, help='path to the preprocessed validation file (Pytorch Geometric Data)')
parser.add_argument('--testfile', type=str, help='path to the preprocessed test file (Pytorch Geometric Data)')
####################################################################

### (2) The following arguments are used when split_ready==False.###
parser.add_argument('--ori_dataset_path', type=str, default="../../datasets/moleculenet/", help='directory of the original csv file (SMILES string)')
parser.add_argument('--pro_dataset_path', type=str, default="../../datasets/moleculenet_pro/", help='directory of the preprocessed data (Pytorch Geometric Data)')
parser.add_argument('--split_mode', type=str, default='random', help=' split methods, use random, stratified or scaffold')
parser.add_argument('--split_train_ratio', type=float, default=0.8, help='the ratio of data for training set')
parser.add_argument('--split_valid_ratio', type=float, default=0.1, help='the ratio of data for validation set')
parser.add_argument('--split_seed', type=int, default=122, help='random seed for split, use 122, 123 or 124')
####################################################################

parser.add_argument('--log_dir', type=str, default=None, help='directory to save train/val information') 
parser.add_argument('--save_dir', type=str, default='../trained_models/your_model/', help='directory to save the model with best validation performance')
parser.add_argument('--evaluate', action='store_true', default=False, help='specify it to be true if you want to do evaluation on test set (available only when test labels are provided)')

args = parser.parse_args()



def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

tab_printer(args)



### Use corresponding configuration for different datasets
sys.path.append("..")
confs = __import__('config.config_'+args.dataset, fromlist=['conf'])
conf = confs.conf
print(conf)




### Get and split dataset
if not args.split_ready:
    dataset, num_node_features, num_edge_features, num_graph_features = get_dataset(args.pro_dataset_path, args.dataset, conf['graph_level_feature'])
    assert conf['num_tasks'] == dataset[0].y.shape[-1]
    train_dataset, val_dataset, test_dataset = split_data(args.ori_dataset_path, args.dataset, dataset, args.split_mode, args.split_seed, split_size=[args.split_train_ratio, args.split_valid_ratio, 1.0-args.split_train_ratio-args.split_valid_ratio])
else:
    train_dataset = torch.load(args.trainfile)
    val_dataset = torch.load(args.validfile)
    test_dataset = torch.load(args.testfile)
    num_node_features = train_dataset[0].x.size(1)
    num_edge_features = train_dataset[-1].edge_attr.size(1)
    num_graph_features = None
    if conf['graph_level_feature']:
        num_graph_features = train_dataset[0].graph_attr.size(-1)
    train_dataset = [JunctionTreeData(**{k: v for k, v in data}) for data in train_dataset]
    val_dataset = [JunctionTreeData(**{k: v for k, v in data}) for data in val_dataset]
    test_dataset = [JunctionTreeData(**{k: v for k, v in data}) for data in test_dataset]
print("======================================")
print("=====Total number of graphs in", args.dataset,":", len(train_dataset)+len(val_dataset)+len(test_dataset), "=====")
print("=====Total number of training graphs in", args.dataset,":", len(train_dataset), "=====")
print("=====Total number of validation graphs in", args.dataset,":", len(val_dataset), "=====")
print("=====Total number of test graphs in", args.dataset,":", len(test_dataset), "=====")
print("======================================")



### Choose model
if conf['model'] == "ml2": ### Multi-level model
    model = MLNet2(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
elif conf['model'] == "ml3": ### ablation studey: w/o subgraph-level
    model = MLNet3(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
else:
    print('Please choose correct model!!!')



### Run
if conf['task_type'] == "classification":
    run_classification(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['early_stopping'], conf['loss'], conf['metric'], args.log_dir, args.save_dir, args.evaluate)
elif conf['task_type'] == "regression":
    run_regression(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['early_stopping'], conf['metric'], args.log_dir, args.save_dir, args.evaluate)
else:
    print("Wrong task type!!!")

