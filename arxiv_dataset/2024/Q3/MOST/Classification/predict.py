import argparse
import logging
import os
import scipy.io
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from Resnet import Resnet
from util.dataset import BasicDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio
def psnr(x,y):
    return peak_signal_noise_ratio(x,y,data_range=np.max(y))

def iou_cal(x,y):
    numer = np.sum(x*y==1)+np.sum(x*y==4)+np.sum(x*y==9)
    denom = np.sum((x>0)+(y>0))
    return numer/denom

def dice_cal(x,y):
    numer = 2 * np.sum(x*y==1)+np.sum(x*y==4)+np.sum(x*y==9)
    denom = np.sum(x>0) + np.sum(y>0)
    return numer/denom

def auc_cal(x,y):

    auc = 0
    TP = 1
    FP = 1
    x_sort = np.sort(x)
    for ind in range(len(x)):
        TP_old = TP
        FP_old = FP
        th = x_sort[ind]
        TP = np.sum([((a>th)  *(b ==1)) for (a,b) in zip(x,y)])/np.sum(y)
        FP = np.sum([((a>th)  *(b==0)) for (a,b) in zip(x,y)])/(len(y)-np.sum(y))
        #auc += np.mean([((a>(np.log(0.01*th/(1-0.01*th))))  *(b==1)) for (a,b) in zip(x,y)])
        auc += (TP_old + TP)*(FP_old-FP)/2
    auc += (TP)*(FP)/2
    return auc

def acc_cal(x,y):
    return np.mean([((a>0)==b) for (a,b) in zip(x,y)])

def predict_net(net,args,val_loader):

    net.eval()
    
    n_val = len(val)
    
    count = 0 
    count_right = 0
    labels = []
    activations = []
    for imgs, label   in tqdm(val_loader):
        imgs = imgs.cuda(args.gpu_ind[0])
        label = label.cuda(args.gpu_ind[0])
        with torch.no_grad():
            activation = nn.parallel.data_parallel(net, imgs, args.gpu_ind)
        
        labels.append(np.squeeze(int(label.cpu().detach().numpy())))
        activations.append(np.squeeze(activation.cpu().detach().numpy()))
        

    
    return acc_cal(activations,labels), auc_cal(activations,labels)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,help='Number of epochs', dest='epochs')
    parser.add_argument('--batch_size', type=int, nargs='?', default=1,help='Batch size', dest='batchsize')
    parser.add_argument('--lr', type=float, nargs='?', default=1e-4,help='Learning rate', dest='lr')
    parser.add_argument('-da', '--dataset', dest='dataset', type=str, default='Task4_IXI-T1_Sex_class',help='dataset') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='0',help='gpu')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    parser.add_argument('--load',  type=int, default=100,help='Checkpoint load')

    return parser.parse_args()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    str_ids = args.gpu_ind.split(',')
    args.gpu_ind = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ind.append(id)

    # set gpu ids
    if len(args.gpu_ind) > 0:
        torch.cuda.set_device(args.gpu_ind[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device CUDA {(args.gpu_ind)}')
    
    if args.dataset == 'Task4_IXI-T1_Sex_class':
        flattened_shape=[-1, 512, 5, 7, 2]
    elif args.dataset.startswith('Task5_ADNI'):
        flattened_shape = [-1, 512, 6, 8, 2]
    net = Resnet(n_classes = 1,flattened_shape=flattened_shape)
    net.cuda()
    chck = f'/home/hwihun/clcl/Classification/checkpoints/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}/epoch_best.pth'
    net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    
    val = BasicDataset(args,'test')
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    acc,auc  = predict_net(net,args,val_loader)
    
    print(f'ACC: {acc}, AUC: {auc}')