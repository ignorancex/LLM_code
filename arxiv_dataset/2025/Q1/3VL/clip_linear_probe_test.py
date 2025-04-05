import argparse
import logging
import math
import os
import yaml
from PIL import Image
import numpy as np
import clip
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageNet
from tqdm import tqdm, trange
import info_nce
from CLIP_linear_probe import LinearProbeClipModel, check_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Evaluate Linear probe CLIP on ImageNet')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--checkpoint_file', default='', type=str,
                    help='filename to save checkpoint to')


def main():
    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    print(f'args.batch_size: {args.batch_size}')
    
    model_name = "ViT-B/32"
    clip_model, preprocess = clip.load(model_name, device=device, jit=False)
    clip_model.float()

    input_size = 512
    num_classes = 1000
    linear_probe_model = LinearProbeClipModel(input_size, num_classes)
    linear_probe_model.to(device)

    filename = 'checkpoint_imgnet_linear_probe_epoch# 0009.pt.tar'

    if args.checkpoint_file:
        filename = args.checkpoint_file

    print(f'loading model state from filename: {filename}')
    checkpoint = torch.load(filename) 
    linear_probe_model.load_state_dict(checkpoint['state_dict'])
    
    imagenet_val_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='val', transform=preprocess)
    #imagenet_test_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='test', transform=preprocess)
    #imagenet_train_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='train', transform=preprocess)

    val_loader = DataLoader(dataset=imagenet_val_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=imagenet_test_dataset, batch_size=batch_size, shuffle=True)
    #train_loader = DataLoader(dataset=imagenet_train_dataset, batch_size=batch_size, shuffle=True)

    save_frequency = 5

    with torch.no_grad():
        accuracy = check_accuracy(val_loader, linear_probe_model, clip_model)
        print(f"Accuracy on ImageNet test set: {accuracy*100:.2f}%")
        
if __name__ == "__main__":
    main()


    


