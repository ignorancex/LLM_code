import argparse
import logging
import os
import scipy.io
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from unet import *
from utils.dataset import BasicDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio
def psnr(x,y):
    return peak_signal_noise_ratio(x,y,data_range=np.max(y))

def iou_cal(x,y):
    numer = np.sum(x*y)
    denom = np.sum(x+y)
    return numer/denom

def dice_cal(x,y):
    numer = 2 * np.sum(x*y)
    denom = np.sum(x) + np.sum(y)
    return numer/denom

def predict_net(net,args,val_loader):

    net.eval()
    
    n_val = len(val)
    
    iou = []
    dice = []
    for imgs, true_masks  in tqdm(val_loader):
        imgs = imgs.cuda(args.gpu_ind[0])
        true_masks = true_masks.cuda(args.gpu_ind[0])
        
        with torch.no_grad():
            mask_pred= nn.parallel.data_parallel(net, imgs, args.gpu_ind)
            
        for i in range(imgs.shape[0]):
            pred = torch.squeeze(mask_pred[i,...]).cpu().detach().numpy() >0
            label = torch.squeeze(true_masks[i,...]).cpu().detach().numpy() >0
            if np.sum(label) > 0:
                iou.append(iou_cal(pred,label))
                dice.append(dice_cal(pred,label))
        
    return iou, dice

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
    parser.add_argument('--batch_size', type=int, nargs='?', default=20,help='Batch size', dest='batchsize')
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3,help='Learning rate', dest='lr')
    parser.add_argument('-da', '--dataset', dest='dataset', type=str, default='Task3_BRATS_Tumor_seg',help='dataset') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='3',help='gpu')
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
    
    net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
    net.cuda()
    
    chck = f'/home/hwihun/clcl/Segmentation/checkpoints/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}/epoch_best.pth'
    net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    
    val = BasicDataset(args,'test')
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    iou, dice  = predict_net(net,args,val_loader)
    print(f'IoU: {np.mean(iou)}, DICE: {np.mean(dice)}')