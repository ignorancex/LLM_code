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

import random
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

def train_net(dcnet,tnet,task_net,memory,device,args):

    train = BasicDataset(args,'train_cl')
    if  args.task.endswith('class'):
        val = BasicDataset(args,'val_task')
    else:
        val = BasicDataset(args,'val_cl')
    n_train = len(train)
    n_val = len(val)
    

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment= f'_Trn_cl_{args.task}_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_LR_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')
    
    
    optimizer_t = optim.Adam(dcnet.parameters(), lr=args.lr)
    
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
                                fimg_val = torch.reshape(fimg_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                                kspace_val = torch.reshape(kspace_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))                     
                
                            dc_output = nn.parallel.data_parallel(dcnet, (kspace_val, kspace_val, mask_val), args.gpu_ind)
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
                        os.makedirs(os.path.join(args.path, f'Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_chk/{args.task}'), exist_ok=True)
                        torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_chk/{args.task}') +f'/dcnet_step_best.pth')
                        val_loss_best = tot_loss

                if global_step % args.save_step == 0:
                    os.makedirs(os.path.join(args.path, f'Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_chk/{args.task}'), exist_ok=True)
                    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_chk/{args.task}') +f'/dcnet_step{global_step}.pth')

            
                dcnet.train()
                imgs = imgs.cuda(args.gpu_ind[0])
                kspace = kspace.cuda(args.gpu_ind[0])
                mask = mask.cuda(args.gpu_ind[0]).byte()
                target= target.cuda(args.gpu_ind[0])
                fimg= fimg.cuda(args.gpu_ind[0])
                 
                if args.task.endswith('class'):
                    img_shape = imgs.shape
                    mask = mask[0:1,:,:,:]
                    imgs = torch.reshape(imgs,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                    kspace = torch.reshape(kspace,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))
                    fimg = torch.reshape(fimg,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))

                imgs_pred = nn.parallel.data_parallel(dcnet, (kspace, kspace, mask), args.gpu_ind)

                
                if args.task.endswith('class'):
                    imgs_pred = torch.reshape(imgs_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)
                else:
                    loss_ssim = ssimloss(imgs_pred,fimg)
                    
                preds = nn.parallel.data_parallel(task_net, imgs_pred, args.gpu_ind)
                
                loss_task = criterion(preds,target)#,data_range=torch.max(imgs_pred,(1,2,3)))
                
                if args.task.endswith('class'):
                    loss = loss_task
                    loss_ssim = loss_task-loss_task
                else:
                    loss = loss_task
                    
                optimizer_t.zero_grad()
                loss.backward()
                optimizer_t.step()   
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train_task', loss_task.item(), global_step)
                writer.add_scalar('Loss/train_ssim', loss_ssim.item(), global_step)
                              
                                                               
                pbar.update(1)
                
                global_step += 1    
                
                if global_step % args.memory_step == 0:
                    dcnet.train()
                    
                    rand_ind = (global_step // args.memory_step) % (len(memory))
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
                                        
                    
                    if memory_task.endswith('class'):
                        imgs_pred = torch.reshape(imgs_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)
                    else:
                        loss_ssim = ssimloss(imgs_pred,fimg_memory)
                        
                    if memory_task.endswith('recon'):
                        loss = loss_ssim
                        loss_task = loss_ssim - loss_ssim
                    else:
                        preds = nn.parallel.data_parallel(task_net_memory, imgs_pred, args.gpu_ind)
                        loss_task = criterion_task(preds,target_memory)#,data_range=torch.max(imgs_pred,(1,2,3)))
                    
                        if memory_task.endswith('class'):
                            loss = loss_task
                            loss_ssim = loss_task-loss_task
                        else:
                            loss = loss_task
                    
                    if args.lr != 0.0001:    
                        loss = ( 0.0001 / args.lr )  * loss   
               
                    optimizer_t.zero_grad()
                    loss.backward()
                    optimizer_t.step()  
                    
                    writer.add_scalar(f'Loss/train_memory_{memory_task}', loss.item(), global_step)
                    writer.add_scalar(f'Loss/train_memory_{memory_task}_task', loss_task.item(), global_step)
                    writer.add_scalar(f'Loss/train_memory_{memory_task}_ssim', loss_ssim.item(), global_step)

                        
                    
    os.makedirs(os.path.join(args.path, f'Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_chk/{args.task}'), exist_ok=True)
    torch.save(dcnet.state_dict(),os.path.join(args.path, f'Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_chk/{args.task}') +f'/dcnet_step_last.pth')
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
    
def memory_save_from_all(task,memory_size,args):
    path_img = f'/fast_storage/hwihun/pkl_clcl/pkl_{task}_train_cl.pklv4'
    assert os.path.isfile(path_img), path_img
    images = []
    with open(path_img, "rb") as f:
        images += pickle.load(f)
    assert len(images) > 0, path_img
    
    rand_ind = np.random.choice(range(len(images)),size = memory_size, replace = False) 
    #rand_ind = range(0,memory_size)
    images_new = []
    for ind in rand_ind:
        images_new.append(images[ind])
    
    os.makedirs( f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/', exist_ok=True)
    path_img_new = f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/pkl_memory_{task}_train_cl.pklv4'    
    with open(path_img_new, 'wb') as f:
        pickle.dump(images_new, f, protocol=4)
        
    if task != 'Task0_Fastmri_recon':
        path_target = f'/fast_storage/hwihun/pkl_clcl/pkl_{task}_train_cl_target.pklv4'
        assert os.path.isfile(path_target), path_target
        targets = []
        with open(path_target, "rb") as f:
            targets += pickle.load(f)
        assert len(targets) > 0, path_target
        targets_new = []
        for ind in rand_ind:
            targets_new.append(targets[ind])
        path_target_new = f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/pkl_memory_{task}_train_cl_target.pklv4'
        with open(path_target_new, 'wb') as f:
            pickle.dump(targets_new, f, protocol=4)


def memory_save_from_memory(task,memory_size,args):
    path_img = f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/pkl_memory_{task}_train_cl.pklv4'    
    assert os.path.isfile(path_img), path_img
    images = []
    with open(path_img, "rb") as f:
        images += pickle.load(f)
    assert len(images) > 0, path_img
    
    rand_ind = np.random.choice(range(len(images)),size = memory_size, replace = False) 
    #rand_ind = range(0,memory_size)
    images_new = []
    for ind in rand_ind:
        images_new.append(images[ind])
    path_img_new = f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/pkl_memory_{task}_train_cl.pklv4'    
    with open(path_img_new, 'wb') as f:
        pickle.dump(images_new, f, protocol=4)
        
    path_target = f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/pkl_memory_{task}_train_cl_target.pklv4'
    assert os.path.isfile(path_target), path_target
    targets = []
    with open(path_target, "rb") as f:
        targets += pickle.load(f)
    assert len(targets) > 0, path_target
    targets_new = []
    for ind in rand_ind:
        targets_new.append(targets[ind])
    path_target_new = f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/pkl_memory_{task}_train_cl_target.pklv4'
    with open(path_target_new, 'wb') as f:
        pickle.dump(targets_new, f, protocol=4)
        
        
        
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,help='Learning rate', dest='lr')
    parser.add_argument('-t', '--task', dest='task', type=str, default='Task5_ADNI_ADCN_class',help='task') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='1',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=20,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=50,help='Checkpoint saving step')
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
    args.name = f'Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}'
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
    args.task_list  = task_list   
    
    
    dcnet = VarNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
    if task_ind == 1:
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_CL_Task0_Fastmri_recon_vn_onlydc_ssimloss_LR_0.001_chk/dcnet_epoch_best.pth'
    else:
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_cl_start_dc_arch_dc_taskloss_baseline_replay{args.memory_size}_step{args.memory_step}_chk/{task_list[task_ind-1]}/dcnet_step_best.pth'
    logging.info(chck)
    dcnet.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    dcnet.cuda(args.gpu_ind[0])

    memory_allocates = 0
    for ind in range(0,task_ind-1):
        if task_list[ind].endswith('recon') or task_list[ind].endswith('seg'):
            memory_save_from_memory(task_list[ind],50*int(args.memory_size/task_ind),args)
        else:
            memory_save_from_memory(task_list[ind],int(args.memory_size/task_ind),args)
        memory_allocates += int(args.memory_size/task_ind)
        
    if task_list[task_ind-1].endswith('recon') or task_list[task_ind-1].endswith('seg'):
        memory_save_from_all(task_list[task_ind-1],50*(args.memory_size-memory_allocates),args)
    else:
        memory_save_from_all(task_list[task_ind-1],args.memory_size-memory_allocates,args)
    
    
    target_replay = []
    if task_list[task_ind-1].endswith('recon'):
        dataset_replay = BasicDataset_recon(args,'train_cl',memory=True,memory_task = task_list[task_ind-1]) 
        task_net = None
    elif task_list[task_ind-1].endswith('seg'):
        dataset_replay = BasicDataset_seg(args,'train_cl',memory=True,memory_task = task_list[task_ind-1])
        direc = 'Segmentation'
        from unet import UNet
        task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
        task_net.to(device=device)
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{task_list[task_ind-1]}_classnum_{args.class_num}_LR_0.001/epoch_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    else:
        dataset_replay = BasicDataset_class(args,'train_cl',memory=True,memory_task = task_list[task_ind-1])
        direc = 'Classification'
        from Resnet import Resnet
        if task_list[task_ind-1] == 'Task4_IXI-T1_Sex_class':
            flattened_shape=[-1, 512, 5, 7, 2]
        elif task_list[task_ind-1] == 'Task5_ADNI_ADCN_class':
            flattened_shape = [-1, 512, 6, 8, 2]
        task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape)
        task_net.to(device=device)
        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{task_list[task_ind-1]}_classnum_{args.class_num}_LR_0.0001/epoch_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    if task_net is not None:
        task_net.eval()
        
        
    data_loader_replay = DataLoader(dataset_replay, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    for imgs_val, kspace_val, mask_val, fimg_val, target_val  in data_loader_replay:
        imgs_val = imgs_val.cuda(args.gpu_ind[0])
        kspace_val = kspace_val.cuda(args.gpu_ind[0])
        mask_val = mask_val.cuda(args.gpu_ind[0]).byte()
        fimg_val= fimg_val.cuda(args.gpu_ind[0])
        target_val= target_val.cuda(args.gpu_ind[0])
        with torch.no_grad():
            if task_list[task_ind-1].endswith('class') or task_list[task_ind-1].endswith('pred'):
                img_shape = imgs_val.shape
                mask_val = mask_val[0:1,:,:,:]
                imgs_val = torch.reshape(imgs_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                fimg_val = torch.reshape(fimg_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                kspace_val = torch.reshape(kspace_val,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))                     

            mask_pred = nn.parallel.data_parallel(dcnet, (kspace_val, kspace_val, mask_val), args.gpu_ind)
            
            if task_list[task_ind-1].endswith('class') or task_list[task_ind-1].endswith('pred'):
                mask_pred = torch.reshape(mask_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)
                
            if task_list[task_ind-1].endswith('recon'):
                preds = mask_pred
            else:
                preds = nn.parallel.data_parallel(task_net, mask_pred, args.gpu_ind)
        if task_list[task_ind-1].endswith('class'):
            target_replay.append(preds[0].cpu().detach().numpy())       
        else:
            target_replay.append(np.transpose(preds[0].cpu().detach().numpy(), [1, 2, 0]))        
    path_target_new = f'/fast_storage/hwihun/pkl_clcl_memory/{args.name}/pkl_memory_{task_list[task_ind-1]}_train_cl_target.pklv4'
    with open(path_target_new, 'wb') as f:
        pickle.dump(target_replay, f, protocol=4)
        
    memory = []
    datasets = []
    args.criterions = []
    args.task_net_memory = []
    for ind in range(0,task_ind):
        print(task_list[ind])
        if task_list[ind].endswith('recon'):
            datasets.append(BasicDataset_recon_wtarget(args,'train_cl',memory=True,memory_task = task_list[ind]) )
            task_net = None
            batchsize = 16
        elif task_list[ind].endswith('seg'):
            datasets.append(BasicDataset_seg_wtarget(args,'train_cl',memory=True,memory_task = task_list[ind]) )
            batchsize = 24
            direc = 'Segmentation'
            from unet import UNet
            task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
            task_net.to(device=device)
            chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{task_list[ind]}_classnum_{args.class_num}_LR_0.001/epoch_best.pth'
            logging.info(chck)
            task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
            
        elif task_list[ind].endswith('class'):
            datasets.append(BasicDataset_class_wtarget(args,'train_cl',memory=True,memory_task = task_list[ind]) )
            direc = 'Classification'
            from Resnet import Resnet
            if task_list[ind] == 'Task4_IXI-T1_Sex_class':
                flattened_shape=[-1, 512, 5, 7, 2]
            elif task_list[ind] == 'Task5_ADNI_ADCN_class':
                flattened_shape = [-1, 512, 6, 8, 2]
            task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape)
            task_net.to(device=device)
            
            chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{task_list[ind]}_classnum_{args.class_num}_LR_0.0001/epoch_best.pth'

            logging.info(chck)
            task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
            batchsize = 1
            
        if task_list[ind].endswith('seg'):
            criterion = nn.MSELoss()
        elif task_list[ind].endswith('class'):
            criterion = torch.nn.MSELoss()
        else:
            criterion = SSIMLoss()
        
        memory.append( DataLoader(datasets[ind], batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True))
        args.criterions.append(criterion)
        if task_net is not None:
            task_net.eval()
        args.task_net_memory.append(task_net)
        
    if args.task.endswith('seg'):
        from util.dataset_vn_cl_seg import BasicDataset
        from unet import UNet
        direc = 'Segmentation'
        task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
        task_net.to(device=device)
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
        task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape)
        #task_net = Resnet(n_classes = args.class_num+1)
        task_net.to(device=device)


        chck = f'/home/hwihun/clcl/{direc}/checkpoints/Trn_downstream_{args.task}_classnum_{args.class_num}_LR_0.0001/epoch_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 1
        

    task_net.eval()
        
    train_net(dcnet = dcnet, tnet = None, task_net = task_net, memory = memory, device = device, args = args)
    
    


