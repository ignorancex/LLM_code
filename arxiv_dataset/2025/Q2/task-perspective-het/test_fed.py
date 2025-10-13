import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datasets import BaseMultiTaskDataset
import clip
import torch
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from copy import deepcopy
from torch.optim import lr_scheduler
from models import HugeModel
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from losses import EdgeLoss
from argparse import ArgumentParser
import utils
from network import SimpleNetwork
import gc

def parse_input():

    parser = ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='../data/taskonomy_dataset/')
    parser.add_argument('--n_rounds', type=int, default=100)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_clients', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--init_lr', type=float, default=1e-1)
    parser.add_argument('--lr_decay_rate', type=float, default=0.999)
    parser.add_argument('--sgd_per_epoch', type=int, default=50)
    parser.add_argument('--epoch_per_round', type=int, default=1)
    parser.add_argument('--min_client_ds_len', type=int, default=128)
    parser.add_argument('--no_improvement_tolerance', type=int, default=5)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--task', type=str)
    parser.add_argument('--method', choices=['fedavg', 'fedprox', 'scaffold', 'fedrep', 'fedamp', 'ditto'], type=str, default='fedavg')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--init_model_dir', type=str, default='./initial_models/')
    parser.add_argument('--trained_model_dir', type=str, default='./trained_models/federated/')
    parser.add_argument('--use_accelerator', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--assignment_files_path', type=str, default='./assignments/')
    parser.add_argument('--type', type=str, choices=['embedding_based', 'class_based', 'lr_test', 'sgd_test'], default='embedding_based')
    parser.add_argument('--start_warm', type=bool, default=False)
    parser.add_argument('--plot_distribution', type=bool, default=False)
    
    return parser.parse_args()


args = parse_input()
path_middle_args = ['test_federated', args.method, args.task, args.type]
current_run_path = utils.create_result_dir(args.result_path, path_middle_args)

np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)

device = utils.get_device(args.use_accelerator)

sample_clip_model, _ = clip.load("ViT-B/32", device=device)

# Create a dataset object
sample_dataset = utils.data.create_global_dataset(args, [args.task])
if args.type == 'class_based':
    assignments = utils.data.create_global_dataset(args, ["class_scene"]).labels["class_scene"]
else:
    assignments = np.load(os.path.join(args.assignment_files_path, f'assignments_{args.task}.npy'))
    

# Instantiate network and clients
sample_network = SimpleNetwork(sample_dataset, assignments, device, args, current_run_path)

# Train federated network
mean_loss_history, std_loss_history = sample_network.train()

plt.figure(figsize=(10, 5))
plt.plot(mean_loss_history)
#plot the std loss as shaded area
plt.fill_between(range(len(mean_loss_history)), np.array(mean_loss_history)-np.array(std_loss_history), np.array(mean_loss_history)+np.array(std_loss_history), alpha=0.5)
plt.title('Global Model Loss')
plt.xlabel('Global Round')
plt.ylabel('Loss')
plt.title("Mean & STD of Validation Loss on All Clients")
plt.savefig(os.path.join(current_run_path, f'eetf_{args.rand_seed}_global_{args.task}_model_loss_{args.alpha}_{args.sgd_per_epoch}.png'))

torch.save(sample_network.global_model.state_dict(), os.path.join(current_run_path, f"eetf_{args.rand_seed}_global_{args.task}_model_{str(args.alpha)}_{args.sgd_per_epoch}.pth"))