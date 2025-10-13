from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# pylint: skip-file
import argparse
import os
import pickle as cp
import torch


cmd_opt = argparse.ArgumentParser(description='Argparser for model runs', allow_abbrev=False)

## Needed Original BiGG Arguments:
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-data_dir', default='.', help='data dir')
cmd_opt.add_argument('-eval_folder', default=None, help='data eval_dir')
cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-g_type', default=None, help='graph type')
cmd_opt.add_argument('-model_dump', default=None, help='load model dump')
cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-num_proc', type=int, default=1, help='number of processes')
cmd_opt.add_argument('-node_order', default='default', help='default/DFS/BFS/degree_descent/degree_accent/k_core/all, or any of them concat by +')
cmd_opt.add_argument('-dist_backend', default='gloo', help='dist package backend', choices=['gloo', 'nccl'])
cmd_opt.add_argument('-embed_dim', default=256, type=int, help='embed size')
cmd_opt.add_argument('-bits_compress', default=0, type=int, help='num of bits to compress')
cmd_opt.add_argument('-param_layers', default=1, type=int, help='num of param groups')
cmd_opt.add_argument('-num_test_gen', default=-1, type=int, help='num of graphs generated for test')
cmd_opt.add_argument('-max_num_nodes', default=-1, type=int, help='max num of nodes')
cmd_opt.add_argument('-rnn_layers', default=1, type=int, help='num layers in rnn')
cmd_opt.add_argument('-seed', default=34, type=int, help='seed')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='gradient clip')
cmd_opt.add_argument('-train_ratio', default=0.7, type=float, help='ratio for training')
cmd_opt.add_argument('-dev_ratio', default=0.1, type=float, help='ratio for dev')
cmd_opt.add_argument('-greedy_frac', default=0, type=float, help='prob for greedy decode')
cmd_opt.add_argument('-num_epochs', default=100000, type=int, help='num epochs')
cmd_opt.add_argument('-batch_size', default=10, type=int, help='batch size')
cmd_opt.add_argument('-pos_enc', default=True, type=eval, help='pos enc?')
cmd_opt.add_argument('-pos_base', default=10000, type=int, help='base of pos enc')
cmd_opt.add_argument('-tree_pos_enc', default=False, type=eval, help='pos enc for tree?')
cmd_opt.add_argument('-blksize', default=-1, type=int, help='num blksize steps')
cmd_opt.add_argument('-accum_grad', default=1, type=int, help='accumulate grad for batching purpose')
cmd_opt.add_argument('-epoch_save', default=50, type=int, help='num epochs between save')
cmd_opt.add_argument('-epoch_load', default=None, type=int, help='epoch for loading')
cmd_opt.add_argument('-batch_exec', default=False, type=eval, help='run with dynamic batching?')
cmd_opt.add_argument('-share_param', default=True, type=eval, help='share param in each level?')
cmd_opt.add_argument('-directed', default=False, type=eval, help='is directed graph?')
cmd_opt.add_argument('-self_loop', default=False, type=eval, help='has self-loop?')
cmd_opt.add_argument('-bfs_permute', default=False, type=eval, help='random permute with bfs?')
cmd_opt.add_argument('-display', default=False, type=eval, help='display progress?')


## Extended BiGG Hyperparameters
cmd_opt.add_argument('-has_edge_feats', default=True, type=eval, help='has edge features?')
cmd_opt.add_argument('-has_node_feats', default=False, type=eval, help='has node features?')


## Synthetic Graph Generation
cmd_opt.add_argument('-num_graphs', default = 1000, type = int, help = "number of graphs to generate")


## ER Hyperparameters
cmd_opt.add_argument('-p_er', default = 0.01, type = float, help = "prob for ER graph")
cmd_opt.add_argument('-min_er_nodes', default = 250, type = int, help = "min number of nodes in ER graph")
cmd_opt.add_argument('-max_er_nodes', default = 750, type = int, help = "max number of nodes in ER graph")


## DB Hyperparameters
cmd_opt.add_argument('-load_db', default = False, type = eval, help = "load db graphs for stats?")


## Tree Hyperparameters
cmd_opt.add_argument('-num_leaves', default=100, type=int, help='number of leaves in tree')
cmd_opt.add_argument('-by_time', default=False, type=bool, help='order tree by time?')
cmd_opt.add_argument('-tree_type', default='sep', type=str, help='weight+top sampling for trees: sep or joint')
cmd_opt.add_argument('-leaf_order', default = 'default', type = str, help = "TREES ONLY: Should leaves be ordered after internal nodes?")
cmd_opt.add_argument('-source', default=-1, type=int, help='node source for ordering')


## Lobster Hyperparameters
cmd_opt.add_argument('-num_lobster_nodes', default=80, type=int, help='mean lobster nodes')
cmd_opt.add_argument('-p1', default=0.7, type=float, help='probability of edge 1 hop away')
cmd_opt.add_argument('-p2', default=0.7, type=float, help='probability of edge 2 hops away')
cmd_opt.add_argument('-min_nodes', default=5, type=int, help='min lobster nodes')
cmd_opt.add_argument('-max_nodes', default=100, type=int, help='max lobster nodes')


## GCN Hyperparameters
cmd_opt.add_argument('-node_embed_dim', default=128, type=int, help='embed size for nodes')
cmd_opt.add_argument('-out_dim', default=128, type=int, help='embed size for GCN')


## Training Hyperparameters
cmd_opt.add_argument('-epoch_plateu', default=-1, type=int, help='when to plateu learning rate in train')
cmd_opt.add_argument('-wt_plateu', default=-1, type=int, help='when to plateu learning rate in topology')
cmd_opt.add_argument('-top_plateu', default=-1, type=int, help='when to plateu learning rate in weights')
cmd_opt.add_argument('-learning_rate_top', default=1e-3, type=float, help='custom learning rate for topology')
cmd_opt.add_argument('-learning_rate_wt', default=1e-3, type=float, help='custom learning rate for weight')
cmd_opt.add_argument('-learning_rate_top_update', default=1e-3, type=float, help='update learning rate for topology mid training')
cmd_opt.add_argument('-learning_rate_wt_update', default=1e-3, type=float, help='update learning rate weight mi training')
cmd_opt.add_argument('-scale_loss', default=1, type=float, help='scalar to divide weight loss by during training')
cmd_opt.add_argument('-schedule', default=True, type=eval, help='allows user to set scale_loss param')
cmd_opt.add_argument('-debug', default=False, type=eval, help='debug model?')
cmd_opt.add_argument('-tune_sigma', default=False, type=eval, help='tune sigma for weight MMD?')


## Weight State Hyperparams
cmd_opt.add_argument('-embed_dim_wt', default=16, type=int, help='embed dim for weights')
cmd_opt.add_argument('-mu_0', default=True, type=eval, help='learn param for global mean of weights?')
cmd_opt.add_argument('-dynam_score', default=True, type=eval, help='z-score global mean and variance params')
cmd_opt.add_argument('-sampling_method', default='softplus', type=str, help='method for sampling weights [softplus, lognormal, gamma]')
cmd_opt.add_argument('-wt_mode', default='score', type=str, help='mode to standardize weights')
cmd_opt.add_argument('-method', default='BiGG-E', type=str, help='BiGG-E or BiGG-MLP')
cmd_opt.add_argument('-wt_drop', default=-1, type=float, help='dropout for weight MLPs. -1 signifies NO dropout')
cmd_opt.add_argument('-use_mlp', default=False, type=eval, help='mlp for weight tree lstm')
cmd_opt.add_argument('-wt_scale', default=1.0, type=float, help='scale weights')


### ADJ-LSTM Hyperparamters
cmd_opt.add_argument('-num_gen', default = 1000, type = int, help = "Number of generated graphs")
cmd_opt.add_argument('-tol', default = 0, type = float, help = "Tolerance in edge sampling")
cmd_opt.add_argument('-weighted', default = True, type = eval, help = "Indicator whether graphs are weighted")
cmd_opt.add_argument('-constant_nodes', default = False, type = eval, help = "Do graphs have same # nodes (e.g., no padding needed)?")
cmd_opt.add_argument('-num_layers', default = 4, type = int, help = "number of LSTM layers")
cmd_opt.add_argument('-hidden_dim', default = 128, type = int, help = "number of hidden units for row/col states")
cmd_opt.add_argument('-mode', default = 'score', type = str, help = "Mode to standardize/normalize weights")
cmd_opt.add_argument('-wt_range', default = 1.0, type = float, help = "range for min/max standardization")
cmd_opt.add_argument('-log_wt', default = False, type = eval, help = "log transform weights for embedding?")
cmd_opt.add_argument('-sm_wt', default = False, type = eval, help = "softminus (exp(logw - 1)) transform weights for embedding?")


### Baseline (Erdos-Renyi) Hyperparameters
cmd_opt.add_argument('-baseline', default = False, type = eval, help = "Run Baseline")
cmd_opt.add_argument('-generator', default = 'Gnp', type = str, help = "Baseline Class")


### Scalability Run
cmd_opt.add_argument('-scale_run', default=False, type=eval, help='scalability run')
cmd_opt.add_argument('-training_time', default=False, type=eval, help='get training times for models')
cmd_opt.add_argument('-base_path', default = '../../../scalability_saves', type = str, help = "base path for model load in scalability runs")

cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

if cmd_args.epoch_load == -1 and not cmd_args.scale_run:
    cmd_args.epoch_load = cmd_args.num_epochs

if cmd_args.epoch_load is not None and cmd_args.model_dump is None:
    cmd_args.model_dump = os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % cmd_args.epoch_load)

print(cmd_args)

def set_device(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        cmd_args.gpu = gpu
        cmd_args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        cmd_args.gpu = -1
        cmd_args.device = torch.device('cpu')
        print('use cpu')
