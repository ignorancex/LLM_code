import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from unet import UNet
from util import *
from loss import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader


dir_checkpoint = 'checkpoints'

def train_net(net,device,args):
    train = BasicDataset(args,'train_task')
    val = BasicDataset(args,'val_task')
    n_train = len(train)
    n_val = len(val)
    

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    loss_best = 19099999
    for epoch in range(args.epochs):
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for imgs,labels in train_loader:
                imgs = imgs.cuda(args.gpu_ind[0])
                labels = labels.cuda(args.gpu_ind[0])
                preds = nn.parallel.data_parallel(net, imgs, args.gpu_ind)
               
                loss = criterion(preds, labels)
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()                               
                pbar.update(len(imgs))
                
                global_step += 1
                
                if global_step >= args.max_iter:
                    os.makedirs(dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}', exist_ok=True)
                    torch.save(net.state_dict(),dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}/last.pth')
                    logging.info(f'Checkpoint last saved ! End of training')
                    break
                
        if global_step >= args.max_iter:
            break
            
        if epoch % args.valid_step == 0:
            net.eval()
            loss_val = 0
            count = 0
            for imgs,labels in val_loader:
                imgs = imgs.cuda(args.gpu_ind[0])
                labels = labels.cuda(args.gpu_ind[0])
                with torch.no_grad():
                    preds =  nn.parallel.data_parallel(net, imgs, args.gpu_ind)
                if count == int(len(val_loader)/2) and epoch % args.save_step == 0:
                    im = (preds.cpu().numpy() >0)
                    im_pred = labels.cpu().detach().numpy()
                    inp = imgs.cpu().detach().numpy()
                    fig1 = plt.figure(1)
                    plt.imshow(np.concatenate((inp[0,0,:,:],im[0,0,:,:],im_pred[0,0,:,:]),axis=1),cmap=plt.get_cmap('gray'),vmin=0, vmax=1.5)
                    writer.add_figure('Images', fig1, epoch)
                count += 1
                loss_val += criterion(preds, labels)
                
            logging.info('Val Loss: {}'.format(loss_val.item()/count))
            writer.add_scalar('Loss/val', loss_val.item()/count, epoch)
            if loss_val < loss_best:
                os.makedirs(dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}', exist_ok=True)
                torch.save(net.state_dict(),dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}/epoch_best.pth')
                logging.info(f'Checkpoint Best @ {epoch} saved !')
                loss_best = loss_val
                
        if epoch % args.save_step == 0:
            os.makedirs(dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}', exist_ok=True)
            torch.save(net.state_dict(),dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}/epoch_{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')
            
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs',  type=int, default=70,help='Number of epochs', dest='epochs')
    parser.add_argument('-mi', '--max_iter', type=int, default=200000,help='Max iteration', dest='max_iter')
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=50,help='Batch size', dest='batchsize')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,help='Learning rate', dest='lr')
    parser.add_argument('-lo', '--load', dest='load', type=str, default=False,help='Load model from a .pth file')
    parser.add_argument('-da', '--dataset', dest='dataset', type=str, default='Task1_OASIS1_Tissue_seg',help='dataset') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='2',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=1,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=10,help='Checkpoint saving step')
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
    logging.info(f'Using device {device}')
    
    net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
    net.apply(init_weights)
    net.to(device=device)

    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')
    
    #if torch.cuda.device_count() > 1:
    #    net = nn.DataParallel(net, output_device=1)
    #    logging.info(f'Using multi-GPU')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
        

    train_net(net=net,device=device,args = args)
