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
from var_models import VarNet, Unet

from loss import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import init_weights, init_weights_zeros
import pickle
from dataset import *
torch.autograd.set_detect_anomaly(True)

def train_net(dcnet,tnet,memory,memory_val,device,args):
   

    writer = SummaryWriter(comment= f'_Trn_oracle_wpretrain_difflr_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Learning rate:   {args.lr}
        Device:          {device.type}
    ''')
    
    
    optimizer_dc = optim.Adam(dcnet.parameters(), lr=args.lr)

    ssimloss = SSIMLoss()
    
    logging.info(sum(p.numel() for p in dcnet.parameters() if p.requires_grad))

    val_loss_best = 100000
    with tqdm(total=20000, desc=f'Step') as pbar:
        for global_step  in range(0,args.epochs):
                
            if global_step % args.valid_step == 0:
                dcnet.eval()
                
                tot_loss = 0
                tot_ssim = 0
                
                for (task_ind,val_loader) in enumerate(memory_val):
                    count = 0
                    task_net = args.task_net_memory[task_ind]
                    criterion  = args.criterions[task_ind]
                    val_loss_task = 0
                    for imgs_val, kspace_val, mask_val, fimg_val, target_val  in val_loader:
                        imgs_val = imgs_val.cuda(args.gpu_ind[0])
                        kspace_val = kspace_val.cuda(args.gpu_ind[0])
                        mask_val = mask_val.cuda(args.gpu_ind[0]).byte()
                        fimg_val= fimg_val.cuda(args.gpu_ind[0])
                        target_val= target_val.cuda(args.gpu_ind[0])
                        with torch.no_grad():
                            if task_ind == 3 or task_ind == 4:
                                img_shape = imgs_val.shape
                                mask_val = mask_val[0:1,:,:,:]
                                imgs_val = torch.reshape(imgs_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                                fimg_val = torch.reshape(fimg_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                                kspace_val = torch.reshape(kspace_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))                     
                
                            dc_output = nn.parallel.data_parallel(dcnet, (kspace_val, kspace_val, mask_val), args.gpu_ind)
                            mask_pred = dc_output
                            
                            if task_ind == 3 or task_ind == 4:
                                mask_pred = torch.reshape(mask_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)
                            
                            if task_ind > 0:
                                preds = nn.parallel.data_parallel(task_net, mask_pred, args.gpu_ind)
                            else:
                                preds = mask_pred
                            
                        if count == 10 and global_step % args.save_step == 0:
                            im = preds.cpu().numpy()
                            im_pred = target_val.cpu().detach().numpy()
                            inp = imgs_val.cpu().detach().numpy()
                            fig1 = plt.figure(1)
                            if args.task.endswith('seg'):
                                plt.imshow(np.concatenate((inp[0,0,:,:],im[0,0,:,:],im_pred[0,0,:,:]),axis=1),cmap=plt.get_cmap('gray'),vmin=0, vmax=1.5)
                            else:
                                plt.imshow(inp[0,0,:,:],cmap=plt.get_cmap('gray'),vmin=0, vmax=1.5)
                            writer.add_figure('Images', fig1, global_step)

                        count += 1
                        
                        val_loss_task += criterion(preds,target_val)
                            
                    writer.add_scalar(f'Loss/val-{task_ind}', val_loss_task/count, global_step)
                    tot_loss += val_loss_task/count/5
                if global_step == 0:
                    loss_first = tot_loss

                writer.add_scalar('Loss/val', tot_loss, global_step)
                writer.add_scalar('Loss/valnorm', tot_loss/loss_first, global_step)
                if tot_loss < val_loss_best:
                    pbar.set_postfix(**{'best step': global_step,'val loss': tot_loss.item()/loss_first.item()})
                    os.makedirs(os.path.join(args.path, f'Trn_oracle_wpretrain_difflr_lr_{args.lr}'), exist_ok=True)
                    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_oracle_wpretrain_difflr_lr_{args.lr}') +f'/dcnet_step_best.pth')
                    val_loss_best = tot_loss



            if global_step % args.save_step == 0:
                os.makedirs(os.path.join(args.path, f'Trn_oracle_wpretrain_difflr_lr_{args.lr}'), exist_ok=True)
                torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_oracle_wpretrain_difflr_lr_{args.lr}') +f'/dcnet_step{global_step}.pth')

        
            dcnet.train()
            rand_ind = (global_step ) % (len(memory))
            memory_task = args.task_list[rand_ind]
            criterion_task  = args.criterions[rand_ind]
            train_loader_memory = memory[rand_ind]
            task_net_memory = args.task_net_memory[rand_ind]
            imgs_memory, kspace_memory, mask_memory, fimg_memory, target_memory = next(iter(train_loader_memory))


            imgs_memory = imgs_memory.cuda(args.gpu_ind[0])
            kspace_memory = kspace_memory.cuda(args.gpu_ind[0])
            mask_memory = mask_memory.cuda(args.gpu_ind[0]).byte()
            target_memory= target_memory.cuda(args.gpu_ind[0])
            fimg_memory= fimg_memory.cuda(args.gpu_ind[0])
            if memory_task.endswith('class'):
                img_shape = imgs_memory.shape
                mask_memory = mask_memory[0:1,:,:,:]
                imgs_memory = torch.reshape(imgs_memory,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                fimg_memory = torch.reshape(fimg_memory,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                kspace_memory = torch.reshape(kspace_memory,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))

            imgs_pred = nn.parallel.data_parallel(dcnet, (kspace_memory, kspace_memory, mask_memory), args.gpu_ind)
            
            optimizer_dc.zero_grad()
                        

            if memory_task.endswith('class'):
                imgs_pred = torch.reshape(imgs_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)

                
            if not memory_task.endswith('recon'):
                preds = nn.parallel.data_parallel(task_net_memory, imgs_pred, args.gpu_ind)

                loss = criterion_task(preds,target_memory)#,data_range=torch.max(imgs_pred,(1,2,3)))
                
            else:
                loss = criterion_task(imgs_pred,target_memory)
            
            
            if memory_task.endswith('seg'):
                loss *= 10
            
            loss.backward()
            optimizer_dc.step()  
            
            writer.add_scalar(f'Loss/train_{memory_task}', loss.item(), global_step)
            
            pbar.update(1)
            

                    

                        
                    
    os.makedirs(os.path.join(args.path, f'Trn_oracle_wpretrain_difflr_lr_{args.lr}'), exist_ok=True)
    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_oracle_wpretrain_difflr_lr_{args.lr}') +f'/dcnet_step_last.pth')
    #logging.info(f'Checkpoint {global_step} saved !')
    
    writer.close()

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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200000,help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,help='Learning rate', dest='lr')
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='1',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=200,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=200,help='Checkpoint saving step')
    parser.add_argument('-ms', '--memory_step', dest='memory_step', type=int, default=3,help='memory step')
    parser.add_argument('--load',  type=int, default=0,help='Checkpoint load')
    parser.add_argument('--memory_size',  type=int, default=10,help='Memory size')
    parser.add_argument('--acceleration', dest='acceleration', type=float, default=4,help='acceleration')
    parser.add_argument('--center_fraction', dest='center_fraction', type=float, default=0.08,help='center_fraction')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    args.path = './checkpoints'
    args.name = f'Trn_oracle_wpretrain_difflr_lr_{args.lr}'
    logging.info(f'----------------Oracle-------------------')
    str_ids = args.gpu_ind.split(',')
    args.gpu_ind = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ind.append(id)

    if len(args.gpu_ind) > 0:
        torch.cuda.set_device(args.gpu_ind[0])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device CUDA {(args.gpu_ind)}')

    task_list = ['Task0_Fastmri_recon','Task1_OASIS1_Tissue_seg','Task3_BRATS_Tumor_seg','Task4_IXI-T1_Sex_class','Task5_ADNI_ADCN_class']

    args.task_list  = task_list
    
    
    memory = []
    memory_val = []
    datasets = []
    datasets_val = []
    args.criterions = []
    args.task_net_memory = []
    for ind in range(0,5):
        print(task_list[ind])
        if task_list[ind].endswith('recon'):
            args.task = task_list[ind]
            datasets.append(BasicDataset_recon(args,'train_cl',memory=False,memory_task = task_list[ind]) )
            datasets_val.append(BasicDataset_recon(args,'val_cl',memory=False,memory_task = task_list[ind]) )
            task_net = None
            batchsize = 16
        elif task_list[ind].endswith('seg'):
            args.task = task_list[ind]
            datasets.append(BasicDataset_seg(args,'train_cl',memory=False,memory_task = task_list[ind]) )
            datasets_val.append(BasicDataset_seg(args,'val_cl',memory=False,memory_task = task_list[ind]) )
            batchsize = 24
            direc = 'Segmentation'
            from unet import UNet
            task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
            task_net.to(device=device)
            task_net.eval()
            chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{task_list[ind]}_classnum_{args.class_num}_LR_0.001/epoch_best.pth'
            logging.info(chck)
            task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
            
        elif task_list[ind].endswith('class'):
            args.task = task_list[ind]
            datasets.append(BasicDataset_class(args,'train_cl',memory=False,memory_task = task_list[ind]) )
            datasets_val.append(BasicDataset_class(args,'val_task',memory=False,memory_task = task_list[ind]) )
            direc = 'Classification'
            from Resnet import Resnet
            if task_list[ind] == 'Task4_IXI-T1_Sex_class':
                flattened_shape=[-1, 512, 5, 7, 2]
            elif task_list[ind] == 'Task5_ADNI_ADCN_class':
                flattened_shape = [-1, 512, 6, 8, 2]
            task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape)
            task_net.to(device=device)
            task_net.eval()
            
            chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{task_list[ind]}_classnum_{args.class_num}_LR_0.0001/epoch_best.pth'

            logging.info(chck)
            task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
            batchsize = 1

            
        if task_list[ind].endswith('seg'):
            criterion = nn.BCEWithLogitsLoss()
        elif task_list[ind].endswith('class'):
            criterion = torch.nn.BCEWithLogitsLoss()
        elif task_list[ind].endswith('pred'):
            criterion = torch.nn.L1Loss()
        else:
            criterion = SSIMLoss()
        args.criterions.append(criterion)
        args.task_net_memory.append(task_net)
        memory.append( DataLoader(datasets[ind], batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True))
        memory_val.append( DataLoader(datasets_val[ind], batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True))
    
    
    
    dcnet = VarNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
    dcnet.apply(init_weights)

    chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_CL_Task0_Fastmri_recon_vn_onlydc_ssimloss_LR_0.001_chk/dcnet_epoch_best.pth'
    dcnet.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    dcnet.cuda(args.gpu_ind[0])
    


        
    train_net(dcnet = dcnet, tnet = None, memory = memory,memory_val = memory_val, device = device, args = args)
    
    


