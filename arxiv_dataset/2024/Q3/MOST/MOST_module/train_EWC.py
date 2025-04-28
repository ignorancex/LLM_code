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
from var_models import VarNet_ewc, Unet
from loss import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import init_weights, init_weights_zeros
from dataset import *
import random
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)

def train_net(dcnet,task_net,device,args):

    train = BasicDataset(args,'train_cl')
    if  args.task.endswith('class'):
        val = BasicDataset(args,'val_task')
    else:
        val = BasicDataset(args,'val_cl')
    n_train = len(train)
    n_val = len(val)
    

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment= f'_Trn_{args.task}_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_LR_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')
    if args.arch == 'dc':
        params = [{'params':dcnet.parameters()}]
    
    optimizer = optim.Adam(params, lr=args.lr)
    if args.task.endswith('seg'):
        criterion = nn.BCEWithLogitsLoss()
    elif args.task.endswith('class'):
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.task.endswith('pred'):
        criterion = torch.nn.L1Loss()
    else:
        criterion = SSIMLoss()
    ssimloss = SSIMLoss()
    
    logging.info(sum(p.numel() for p in dcnet.parameters() if p.requires_grad))

    val_loss_best = 100000
    with tqdm(total=len(train_loader)*args.epochs, desc=f'Step') as pbar:
        for epoch in range(args.load,args.epochs):
            for imgs, kspace, mask, fimg, target  in train_loader:
                    
                if global_step % args.valid_step == 0:
                    dcnet.eval()

                    n_val = len(val_loader)  # the number of batch
                    tot_loss = 0
                    ssim_loss = 0
                    count = 0
                    for imgs_val, kspace_val, mask_val, fimg_val, target_val  in val_loader:
                        imgs_val = imgs_val.cuda(args.gpu_ind[0])
                        kspace_val = kspace_val.cuda(args.gpu_ind[0])
                        mask_val = mask_val.cuda(args.gpu_ind[0]).byte()
                        fimg_val= fimg_val.cuda(args.gpu_ind[0])
                        target_val= target_val.cuda(args.gpu_ind[0])
                        with torch.no_grad():
                            if args.task.endswith('class') or args.task.endswith('pred'):
                                img_shape = imgs_val.shape
                                mask_val = mask_val[0:1,:,:,:]
                                imgs_val = torch.reshape(imgs_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                                kspace_val = torch.reshape(kspace_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))                     
                            
                            dc_output = nn.parallel.data_parallel(dcnet, (kspace_val, kspace_val, mask_val), args.gpu_ind)
                            if args.arch == 'dc': 
                                mask_pred = dc_output
                            
                            if args.task.endswith('class') or args.task.endswith('pred'):
                                mask_pred = torch.reshape(mask_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)
                            preds = nn.parallel.data_parallel(task_net, mask_pred, args.gpu_ind)
                            
                        if count == int(n_val/2) and global_step % args.save_step == 0:
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
                        tot_loss += criterion(preds,target_val)
                        if global_step == 0:
                            loss_first = tot_loss

                    writer.add_scalar('Loss/val', tot_loss/count, global_step)
                    writer.add_scalar('Loss/valnorm', tot_loss/loss_first, global_step)
                    if tot_loss < val_loss_best:
                        pbar.set_postfix(**{'best step': global_step,'val loss': tot_loss.item()/loss_first.item()})
                        os.makedirs(os.path.join(args.path, f'Trn_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_chk/{args.task}'), exist_ok=True)
                        torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_chk/{args.task}') +f'/dcnet_step_best.pth')
                        val_loss_best = tot_loss

                if global_step % args.save_step == 0:
                    os.makedirs(os.path.join(args.path, f'Trn_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_chk/{args.task}'), exist_ok=True)
                    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_chk/{args.task}') +f'/dcnet_step{global_step}.pth')
            
                dcnet.train()
                imgs = imgs.cuda(args.gpu_ind[0])
                kspace = kspace.cuda(args.gpu_ind[0])
                mask = mask.cuda(args.gpu_ind[0]).byte()
                target= target.cuda(args.gpu_ind[0])
                fimg= fimg.cuda(args.gpu_ind[0])
                if args.task.endswith('class') or args.task.endswith('pred'):
                    img_shape = imgs.shape
                    mask = mask[0:1,:,:,:]
                    imgs = torch.reshape(imgs,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                    kspace = torch.reshape(kspace,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))

                imgs_pred = nn.parallel.data_parallel(dcnet, (kspace, kspace, mask), args.gpu_ind)

                if args.task.endswith('class') or args.task.endswith('pred'):
                    imgs_pred = torch.reshape(imgs_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)
                else:
                    loss_ssim = ssimloss(imgs_pred,fimg)
                preds = nn.parallel.data_parallel(task_net, imgs_pred, args.gpu_ind)
                
                loss_ewc = ewc_loss(dcnet, args.lamda, True)
                
                loss_task = criterion(preds,target)#,data_range=torch.max(imgs_pred,(1,2,3)))
                
                
                if args.task.endswith('class') or args.task.endswith('pred'):
                    loss = loss_ewc + loss_task
                    loss_ssim = loss_task-loss_task
                else:
                    loss = loss_ewc +  loss_task 
                    
                optimizer.zero_grad()
                loss_task.backward()
                optimizer.step()   
                
                loss =  loss_task
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train_task', loss_task.item(), global_step)
                writer.add_scalar('Loss/train_ewc', loss_ewc.item(), global_step)
                writer.add_scalar('Loss/train_ssim', loss_ssim.item(), global_step)             
                
                pbar.update(1)
                
                global_step += 1    
            
    os.makedirs(os.path.join(args.path, f'Trn_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_chk/{args.task}'), exist_ok=True)
    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_chk/{args.task}') +f'/dcnet_step_last.pth')
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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5,help='Learning rate', dest='lr')

    parser.add_argument('-la', '--lambda',  type=float, default=15,help='lambda', dest='lamda')
    parser.add_argument('-da', '--dataset', dest='dataset', type=str, default='Task0_Fastmri_recon',help='dataset') 
    parser.add_argument('-st', '--start', dest='start', type=str, default='dc',help='start') 
    parser.add_argument('-ar', '--arch', dest='arch', type=str, default='dc',help='start') 
    parser.add_argument('-t', '--task', dest='task', type=str, default='Task5_ADNI_ADCN_class',help='task') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='1',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=20,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=50,help='Checkpoint saving step')
    parser.add_argument('--load',  type=int, default=0,help='Checkpoint load')
    parser.add_argument('--acceleration', dest='acceleration', type=float, default=4,help='acceleration')
    parser.add_argument('--center_fraction', dest='center_fraction', type=float, default=0.08,help='center_fraction')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    args.path = './checkpoints'
    logging.info(f'----------------{args.task}-------------------')
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
    task_ind = task_list.index(args.task)
    
    dcnet = VarNet_ewc(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
    if task_ind == 1:
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_CL_Task0_Fastmri_recon_vn_onlydc_ssimloss_LR_0.001_chk/dcnet_epoch_best.pth'
    else:
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_cl_ewc_start_{args.start}_arch_{args.arch}_lambda_{args.lamda}_chk/{task_list[task_ind-1]}/dcnet_step_best.pth'
    logging.info(chck)
    model_dict = dcnet.state_dict() # 현재 신경망 상태 로드 
    pretrained_dict = torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}'))  # pretrained 상태 로드

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    dcnet.load_state_dict(pretrained_dict)
    #dcnet.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    dcnet.cuda(args.gpu_ind[0])
    
    task_buffer = args.task
    args.task = task_list[task_ind-1]
    
    
    if args.task.endswith('recon'):
        dataset_ewc = BasicDataset_recon(args,'train_cl')
        task_net_ewc = None
        batchsize = 16
    elif args.task.endswith('seg'):
        dataset_ewc = BasicDataset_seg(args,'train_cl')
        batchsize = 24
        direc = 'Segmentation'
        from unet import UNet
        task_net_ewc = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
        task_net_ewc.to(device=device)
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{args.task}_classnum_{args.class_num}_LR_0.001/epoch_best.pth'
        logging.info(chck)
        task_net_ewc.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        
    elif args.task.endswith('class'):
        dataset_ewc = BasicDataset_class(args,'train_cl')
        direc = 'Classification'
        from Resnet import Resnet
        if args.task == 'Task4_IXI-T1_Sex_class':
            flattened_shape=[-1, 512, 5, 7, 2]
        elif args.task == 'Task5_ADNI_ADCN_class':
            flattened_shape = [-1, 512, 6, 8, 2]
        task_net_ewc = Resnet(n_classes = 1,flattened_shape=flattened_shape)
        task_net_ewc.to(device=device)
        
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{args.task}_classnum_{args.class_num}_LR_0.0001/epoch_best.pth'

        logging.info(chck)
        task_net_ewc.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        batchsize = 1
        
    if task_net_ewc is not None:
        task_net_ewc.eval()
        
    if args.task.endswith('seg'):
        criterion_ewc = nn.BCEWithLogitsLoss()
    elif args.task.endswith('class'):
        criterion_ewc = torch.nn.BCEWithLogitsLoss()
    elif args.task.endswith('recon'):
        criterion_ewc = SSIMLoss()
    ewc_loader = DataLoader(dataset_ewc, batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True)
            
    fisher_estimation_sample_size = 1024
    dcnet.consolidate(dcnet.estimate_fisher(ewc_loader, fisher_estimation_sample_size,task_net_ewc,criterion_ewc,batchsize))
    args.task = task_buffer

    
    if args.task.endswith('seg'):
        from util.dataset_vn_cl_seg import BasicDataset
        from unet import UNet
        direc = 'Segmentation'
        task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
        task_net .to(device=device)
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{args.task}_classnum_{args.class_num}_LR_0.001/epoch_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 32
         
    elif args.task.endswith('class'):
        from util.dataset_vn_cl_class import BasicDataset
        from Resnet import Resnet
        direc = 'Classification'
        if args.task == 'Task4_IXI-T1_Sex_class':
            flattened_shape=[-1, 512, 5, 7, 2]
        elif args.task == 'Task5_ADNI_ADCN_class':
            flattened_shape = [-1, 512, 6, 8, 2]
        task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape,dropout = 0)
        
        task_net.to(device=device)
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{args.task}_classnum_{args.class_num}_LR_0.0001/epoch_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 1
        
    elif args.task.endswith('pred'):
        from util.dataset_vn_cl_pred import BasicDataset
        from Resnet import Resnet
        direc = 'Regression'
        flattened_shape = [-1, 512, 6, 7, 2]
        task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape,dropout = 0)
        task_net.to(device=device)
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{args.task}_classnum_{args.class_num}_LR_1e-05/epoch_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 1
    task_net.eval()
        
    train_net(dcnet = dcnet, task_net = task_net, device = device, args = args)


