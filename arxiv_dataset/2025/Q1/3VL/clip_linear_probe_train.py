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

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Linear probe CLIP training on ImageNet')

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
                                     
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--num-cycles', default=10, type=int)

# options for ANCOR
#parser.add_argument('--mode', type=str, required=True,
#                    choices=['fine', 'coarse'],
#                    help="Whether to train with coarse or fine labels")
#parser.add_argument('--queue', type=str, choices=QUEUES, default='multi',
#                    help="Queue type")
#parser.add_argument('--metric', type=str, choices=METRICS, default='angular',
#                    help='Which metric to apply before calculating contrastive dot products')
#parser.add_argument('--calc-types', nargs='*', type=str, choices=CALCULATORS,
#                    default=['cls', 'cst_by_class'],
#                    help='List of loss calculators to be used in training')
#parser.add_argument('--loss-types', nargs='*', type=str, choices=LOSSES,
#                    default=['ce', 'ce'],
#                    help='List of loss methods that receive the logits and labels as inputs')
#parser.add_argument('--head', type=str, default='seq', choices=HEADS)
parser.add_argument('-s', '--save-freq', default=1, type=int,
                    help='save once in how many epochs')
parser.add_argument('--dataset', default='living17',
                    choices=['tiered', 'cifar100', 'living17', 'entity13', 'nonliving26', 'entity30'])
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multistep learning rate scheduler factor.')
parser.add_argument('--keep-epochs', default=[59, 99, 159, 199], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--split', default=None, type=str, choices=['good', 'bad'])


def main():
    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}')
    print(f'args.batch_size: {args.batch_size}')
    
    
   #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

    # Load clip model
    #model_name = "ViT-L/14@336px"
    model_name = "ViT-B/32"
    clip_model, preprocess = clip.load(model_name, device=device, jit=False)
    #clip_model.float()
    input_size = 512
    num_classes = 1000
    linear_probe_model = LinearProbeClipModel(input_size, num_classes)
    linear_probe_model.to(device)


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
    
    #imagenet_val_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='val', transform=preprocess)
    #imagenet_test_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='test', transform=preprocess)
    imagenet_train_dataset = ImageNet(root="/disk5/dataset/ImageNet", split='train', transform=preprocess)

    #val_loader = DataLoader(dataset=imagenet_val_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=imagenet_test_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(dataset=imagenet_train_dataset, batch_size=batch_size, shuffle=True)

    save_frequency = 1

    #params = [p for p in model.parameters() if p.requires_grad]
    #print(f'len(params) : {len(params)}')
    #params = [p for p in parallel_net.parameters() if p.requires_grad]
    #learning_rate = 3*10-6
    
    learning_rate = 3e-6
    weight_decay = 0.1
    #optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    params = linear_probe_model.parameters()
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    #optimizer = torch.optim.Adam(params, lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    #Params used in clip paper, the lr is smaller, more safe for fine tuning to new dataset
    #optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 

    #optimizer.load_state_dict(checkpoint['optimizer'])
       
    # optionally resume from a checkpoint
    start_epoch = 0
    checkpoint_path = args.checkpoint_file 
    if not checkpoint_path:
        checkpoint_path = 'checkpoint_linear_probe_epoch_'


    linear_probe_train(linear_probe_model, clip_model, num_epochs, train_loader, criterion, optimizer, save_frequency, checkpoint_path)

    """tensor board
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    """ 
        
if __name__ == "__main__":
    main()


    


