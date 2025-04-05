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
from CLIP_linear_probe import LinearProbeClipModel, linear_probe_train
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Create a CSV of CLIP encodes images and traget values')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--checkpoint_file', default='', type=str,
                    help='filename to save checkpoint to')


parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:2375', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for ei ther single node or '
                         'multi node data parallel training')
                                     

def main():
    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    """ num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}') """
    print(f'args.batch_size: {args.batch_size}')
    
    
   #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

    # Load clip model
    #model_name = "ViT-L/14@336px"
    model_name = "ViT-B/32"
    clip_model, preprocess = clip.load(model_name, device=device, jit=False)   

    clip_model.eval(

    )
    #freeze params to make sure clip_model is not learning
    for param in clip_model.parameters():
        param.requires_grad = False

    clip_model.float()

    #parallel_net  = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    #parallel_net = parallel_net.to(0)
    #torch.distributed.init_process_group()
    #net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
    #model.cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    imagenet_val_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='val', transform=preprocess)
    #imagenet_test_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='test', transform=preprocess)
    #imagenet_train_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='train', transform=preprocess)

    val_loader = DataLoader(dataset=imagenet_val_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=imagenet_test_dataset, batch_size=batch_size, shuffle=True)
    #train_loader = DataLoader(dataset=imagenet_train_dataset, batch_size=batch_size, shuffle=True)

    #dataset_dict = {
    #    'Features': [],
    #    'Label': []
    #}

    total_data = None
    #for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
      images = images.to(device=device)
      #labels = labels.to(device=device)

      image_features = clip_model.encode_image(images)
      image_features /= image_features.norm(dim=-1, keepdim=True)
      #image_features.float()

      image_features = image_features.to(device='cpu')
      labels = labels.to(device='cpu')
      image_features = image_features.numpy()
      labels = labels.numpy()

      
      #data = zip(image_features, labels)
      labels = labels.reshape(-1, 1)
      #print(f'image_features.shape: {image_features.shape}, labels.shape: {labels.shape}')  
      curr_data = np.concatenate((image_features, labels), axis=1)
      #print(f'\n\ndata: \n{data}')
      if 0 == batch_idx:
        total_data = curr_data
      else:
        total_data = np.concatenate((total_data, curr_data), axis=0)
         
    num_features = 512
    columns = [f'feat{i+1}' for i in range(num_features)]
    #print(f'columns: {columns}')
    columns.append('label')
      #print(f'columns: {columns}')

    frame = pd.DataFrame(data=total_data, columns=columns)
    #print(frame)
    #dataset_dict['Features'].append(image_features)
    #dataset_dict['Label'].append(labels)

    #df = pd.DataFrame(dataset_dict)
    #print (df)
      
    filename = 'clip_imagenet_test.csv'
    frame.to_csv(filename,float_format='%.6f',index=False)#,index=False,header=False
        
   
        
if __name__ == "__main__":
    main()


    


