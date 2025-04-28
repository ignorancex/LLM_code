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
from var_models import VarNet_par, Unet, VarNet
from loss import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import init_weights, init_weights_zeros
import copy
import fastmri

torch.autograd.set_detect_anomaly(True)

def train_net(dcnet,tnets,task_net,device,args):

    train = BasicDataset(args,'train_cl')
    if  args.task.endswith('class'):
        val = BasicDataset(args,'val_task')
    else:
        val = BasicDataset(args,'val_cl')
    n_train = len(train)
    n_val = len(val)
    

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment= f'_Trn_{args.task}_cl_lwf_strat_dc_arch_dc_LR_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    params = [{'params':tnet.parameters()} for tnet in tnets]
    params.append({'params':dcnet.parameters()})
    
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
    tnets_old = [copy.deepcopy(tnet) for tnet in tnets]
    
    with tqdm(total=len(train_loader)*args.epochs, desc=f'Step') as pbar:
        for epoch in range(args.load,args.epochs):
            for imgs, kspace, mask, fimg, target  in train_loader:
                    
                if global_step % args.valid_step == 0:
                    dcnet.eval()
                    for tnet in tnets:
                        tnet.eval() 

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
                            
                            #dc_output = nn.parallel.data_parallel(dcnet, (kspace_val, kspace_val, mask_val), args.gpu_ind)
                            masked_kspace =kspace_val
                            for cascade in dcnet:
                                kspace_val = cascade(kspace_val, masked_kspace, mask_val,args.gpu_ind)
                            kspace_pred =tnets[-1](kspace_val, masked_kspace, mask_val,args.gpu_ind) 
                            #nn.parallel.data_parallel(tnets[-1], kspace_val, args.gpu_ind)
                            mask_pred = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
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
                        os.makedirs(os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}'), exist_ok=True)
                        torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}') +f'/dcnet_step_best.pth')
                        for ind in range(len(tnets)):
                            torch.save(tnets[ind].state_dict(),os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}') +f'/tnet_{ind}_step_best.pth')
                        val_loss_best = tot_loss

                if global_step % args.save_step == 0:
                    os.makedirs(os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}'), exist_ok=True)
                    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}') +f'/dcnet_step{global_step}.pth')
                    
                    for ind in range(len(tnets)):
                        torch.save(tnets[ind].state_dict(),os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}') +f'/tnet_{ind}_step{global_step}.pth')
            
                dcnet.train()
                for tnet in tnets:
                    tnet.train() 
                imgs = imgs.cuda(args.gpu_ind[0])
                kspace = kspace.cuda(args.gpu_ind[0])
                mask = mask.cuda(args.gpu_ind[0]).byte()
                target= target.cuda(args.gpu_ind[0])
                if args.task.endswith('class') or args.task.endswith('pred'):
                    img_shape = imgs.shape
                    mask = mask[0:1,:,:,:]
                    imgs = torch.reshape(imgs,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                    kspace = torch.reshape(kspace,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))
                    
                masked_kspace =kspace
                for cascade in dcnet:
                    kspace = cascade(kspace, masked_kspace, mask,args.gpu_ind)

                loss_lwf = 0
                try:
                    for ind in range(len(tnets)-1):
                        with torch.no_grad():
                            kspace_pred_old = tnets_old[ind](kspace, masked_kspace, mask,args.gpu_ind) 
                            imgs_pred_old = fastmri.complex_abs(fastmri.ifft2c(kspace_pred_old))
                        kspace_pred = tnets[ind](kspace, masked_kspace, mask,args.gpu_ind) 
                        imgs_pred = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
                        loss_lwf += ssimloss(imgs_pred,imgs_pred_old)/(len(tnets)-1)
                except:
                    continue
                    
                kspace_pred =tnets[-1](kspace, masked_kspace, mask,args.gpu_ind) 
                imgs_pred = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
                
                if args.task.endswith('class'):
                    imgs_pred = torch.reshape(imgs_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)

                preds = nn.parallel.data_parallel(task_net, imgs_pred, args.gpu_ind)
                
                loss_task = criterion(preds,target)#,data_range=torch.max(imgs_pred,(1,2,3)))

                loss = args.lamda * loss_lwf + loss_task
                
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train_task', loss_task.item(), global_step)
                writer.add_scalar('Loss/train_lwf', loss_lwf.item(), global_step)
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()                               
                pbar.update(1)
                
                global_step += 1    
            
    os.makedirs(os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}'), exist_ok=True)
    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}') +f'/dcnet_step_last.pth')
    for ind in range(len(tnets)):
        torch.save(tnets[ind].state_dict(),os.path.join(args.path, f'Trn_cl_lwf_strat_dc_arch_dc_chk/{args.task}') +f'/tnet_{ind}_step_last.pth')
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
    parser.add_argument('-da', '--dataset', dest='dataset', type=str, default='Task0_Fastmri_recon',help='dataset') 
    parser.add_argument('-st', '--start', dest='start', type=str, default='dc',help='start') 
    parser.add_argument('-ar', '--arch', dest='arch', type=str, default='dc',help='start') 
    parser.add_argument('-t', '--task', dest='task', type=str, default='Task5_ADNI_ADCN_class',help='task') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='1,3',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=50,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=50,help='Checkpoint saving step')
    parser.add_argument('--load',  type=int, default=0,help='Checkpoint load')
    parser.add_argument('--acceleration', dest='acceleration', type=float, default=4,help='acceleration')
    parser.add_argument('--center_fraction', dest='center_fraction', type=float, default=0.08,help='center_fraction')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    parser.add_argument('-la', '--lambda',  type=float, default=1,help='lambda', dest='lamda')
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
    '''
    dcnet_all = VarNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
    if task_ind == 1:
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_CL_Task0_Fastmri_recon_vn_onlydc_ssimloss_LR_0.001_chk/dcnet_epoch_best.pth'
    else:
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_cl_lwf_strat_dc_arch_dc_chk/{task_list[task_ind-1]}/dcnet_step_best.pth'
    logging.info(chck)
    dcnet_all.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    dcnet_all.cuda(args.gpu_ind[0])
    '''

    dcnet_all = VarNet_par(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
    chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_CL_Task0_Fastmri_recon_vn_onlydc_ssimloss_LR_0.001_chk/dcnet_epoch_best.pth'
    dcnet_all.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    dcnet_all.cuda(args.gpu_ind[0])

    tnets = []
    if task_ind ==1:
        for ind in range(0,task_ind+1):
            if ind == 0:
                dcnet = dcnet_all.cascades[0:-1]
                dcnet.cuda(args.gpu_ind[0])
                tnet = dcnet_all.cascades[-1]
                tnet.cuda(args.gpu_ind[0])
            else:
                tnet = copy.deepcopy(tnet)
                tnet.apply(init_weights)
                                
            tnet.cuda(args.gpu_ind[0])
            tnets.append(tnet)
    else:
        for ind in range(0,task_ind+1):
            #tnet = Unet(in_chans=1, out_chans=1, chans=2)
            dcnet = dcnet_all.cascades[0:-1]
            dcnet.cuda(args.gpu_ind[0])
            chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_cl_lwf_strat_dc_arch_dc_chk/{task_list[task_ind-1]}/dcnet_step_best.pth'
            logging.info(chck)
            dcnet.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
            tnet = dcnet_all.cascades[-1]
            if ind < task_ind:
                tnet = copy.deepcopy(tnet)
                chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_cl_lwf_strat_dc_arch_dc_chk/{task_list[task_ind-1]}/tnet_{ind}_step_best.pth'
                logging.info(chck)
                tnet.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
            else:
                tnet = copy.deepcopy(tnet)
                tnet.apply(init_weights)
            
            tnet.cuda(args.gpu_ind[0])
            tnets.append(tnet)

    if args.task.endswith('seg'):
        from util.dataset_vn_cl_seg import BasicDataset
        from unet import UNet
        direc = 'Segmentation'
        task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
        task_net .to(device=device)
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{args.task}_classnum_{args.class_num}_LR_0.001/epoch_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        task_net.eval()
        args.batchsize = 24
         
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
        task_net.eval()
        args.batchsize = 1

        
    train_net(dcnet = dcnet, tnets = tnets, task_net = task_net, device = device, args = args)


