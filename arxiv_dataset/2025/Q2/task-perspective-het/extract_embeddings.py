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
from models import HugeModel, HugeModelCBAM
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
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--task', type=str)
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--load_model_dir', type=str, default='./trained_models/')
    parser.add_argument('--observe_performance', type=bool, default=False)
    parser.add_argument('--use_accelerator', type=bool, default=False)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--from_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_epoch', type=int, default=0)
    parser.add_argument('--use_cbam', type=bool, default=False)
    
    return parser.parse_args()

args = parse_input()
path_middle_args = ['extract_embeddings', args.task]
current_run_path = utils.create_result_dir(args.result_path, path_middle_args)

np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)

device = utils.get_device(args.use_accelerator)

sample_clip_model, _ = clip.load("ViT-B/32", device=device)

# Create a dataset object
sample_dataset = utils.data.create_global_dataset(args, [args.task])
sample_dataloader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=False)

if args.use_cbam:
    test_model = HugeModelCBAM(sample_clip_model, args.task)
else:
    test_model = HugeModel(sample_clip_model, args.task)
if args.from_checkpoint:
    if args.checkpoint_epoch == 0:
        model_load_path = os.path.join(args.load_model_dir, f"centralized_{args.task}_training{'_cbam' if args.use_cbam else None}_checkpoint.pth")
    else:
        model_load_path = os.path.join(args.load_model_dir, f"centralized_{args.task}_training{'_cbam' if args.use_cbam else None}_checkpoint_{args.checkpoint_epoch}.pth")
    assert os.path.exists(model_load_path), f"Checkpoint file not found at {model_load_path}"
    loaded_chkpt = torch.load(model_load_path, map_location=torch.device("cpu"))
    test_model.load_state_dict(loaded_chkpt['model_state_dict'])
else:
    model_load_path = os.path.join(args.load_model_dir, f"{args.task}_model_clip{'_cbam' if args.use_cbam else None}_latest.pth")
    assert os.path.exists(model_load_path), f"Model file not found at {model_load_path}"
    test_model.load_state_dict(torch.load(model_load_path, map_location=torch.device("cpu")))
test_model = test_model.to(device)


# pass each datapoint through the network and get the embedding
embeddings, test_outputs, test_targets = utils.extract_embeddings(test_model, sample_dataloader, device, args.task, test_mode=False)


# Flatten all embeddings
embeddings = embeddings.view(embeddings.size(0), -1)
# Clear Memory
torch.cuda.empty_cache()
gc.collect()

# Perform k-means clustering on the embeddings
print("Performing k-means clustering")
kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.rand_seed, init="k-means++").fit(embeddings.cpu().numpy())
# Get the cluster centers
centers = kmeans.cluster_centers_
# Get the cluster assignments
assignments = kmeans.labels_
print("Saving the centers and assignments")
# Save the centers to a file
np.save(os.path.join(current_run_path, f'centers_{args.task}.npy'), centers)
# Save the assignments to a file
np.save(os.path.join(current_run_path, f'assignments_{args.task}.npy'), assignments)

if args.observe_performance:
    if args.task == "class_scene":
        utils.plot_confusion_matrix(test_outputs, test_targets, args.task, current_run_path)
    elif args.task == "segment_semantic":
        utils.plot_outputs_segment(test_outputs, test_targets, args.task, current_run_path)
    else:
        utils.plot_outputs(test_outputs, test_targets, args.task, current_run_path)

# Plot TSNE of the embeddings
print("Plotting TSNE of the embeddings")
utils.plot_tsne(embeddings.cpu().numpy(), assignments, args.task, current_run_path, test_targets)
