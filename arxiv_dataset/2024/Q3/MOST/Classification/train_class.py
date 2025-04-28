import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from util.dataset import BasicDataset
from torch.utils.data import DataLoader
import copy
from utils import init_weights
import hdf5storage
from Resnet import Resnet

dir_checkpoint = './checkpoints/'
def auc_cal(x,y):
    auc = 0
    TP = 1
    FP = 1
    for th in range(1,100):
        TP_old = TP
        FP_old = FP
        TP = np.sum([((a>(np.log(0.01*th/(1-0.01*th))))  *(b==1)) for (a,b) in zip(x,y)])/np.sum(y)
        FP = np.sum([((a>(np.log(0.01*th/(1-0.01*th))))  *(b==0)) for (a,b) in zip(x,y)])/(len(y)-np.sum(y))
        #auc += np.mean([((a>(np.log(0.01*th/(1-0.01*th))))  *(b==1)) for (a,b) in zip(x,y)])
        auc += (TP_old + TP)*(FP_old-FP)/2
    return auc

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,help='Learning rate', dest='lr')
    parser.add_argument('-da', '--dataset', dest='dataset', type=str, default='Task5_ADNI_ADCN_class',help='dataset') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='0',help='gpu')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=1,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=5,help='Checkpoint saving step')
    parser.add_argument('-lo', '--load', dest='load', type=str, default=False,help='Load model from a .pth file')
    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    args.path = './searchs'

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
    net.apply(init_weights)
    net.cuda(args.gpu_ind[0])

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
        
    train = BasicDataset(args,'train_task')
    val = BasicDataset(args,'test')
    n_train = len(train)   

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_best = 0
    for epoch in range(args.load,args.epochs):
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for imgs, label  in train_loader:
                imgs = imgs.cuda(args.gpu_ind[0])
                label = label.cuda(args.gpu_ind[0])
                train_output = nn.parallel.data_parallel(net, imgs, args.gpu_ind)
                
                loss = criterion(train_output,label)

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()                               
                pbar.update(imgs.shape[0])
                
                global_step += 1    

            
        if epoch % args.valid_step == 0:
            net.eval()
            valloss = 0
            count = 0
            count_right = 0
            labels = []
            activations = []
            for imgs, label   in val_loader:
                imgs = imgs.cuda(args.gpu_ind[0])
                label = label.cuda(args.gpu_ind[0])
                with torch.no_grad():
                    activation = nn.parallel.data_parallel(net, imgs, args.gpu_ind)
                valloss += criterion(activation,label)
                labels.append(label.cpu().detach().numpy())
                activations.append(activation.cpu().detach().numpy())
                count += 1
            auc_val = auc_cal(activations,labels)
            
            logging.info('Val Loss: {}'.format(valloss/count))
            writer.add_scalar('Loss/val', valloss/count, epoch)
            logging.info('Val AUC: {}'.format(auc_val))
            writer.add_scalar('Loss/val_AUC', auc_val, epoch)
            
            if auc_val > loss_best:
                os.makedirs(dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}', exist_ok=True)
                torch.save(net.state_dict(),dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}/epoch_best.pth')
                logging.info(f'Checkpoint Best @ {epoch} saved !')
                loss_best = auc_val
                
            
        if epoch % args.save_step == 0:
            os.makedirs(dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}', exist_ok=True)
            torch.save(net.state_dict(),dir_checkpoint + f'/Trn_downstream_{args.dataset}_classnum_{args.class_num}_LR_{args.lr}/epoch_{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')

    writer.close()


