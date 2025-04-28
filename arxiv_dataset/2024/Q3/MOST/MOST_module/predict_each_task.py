import argparse
import logging
import os
import scipy.io
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from unet import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
from utils import  init_weights_zeros
from var_models import VarNet, Unet
import pickle


def predict_net(task_net,args,val_loader):

    task_net.eval()
    
    n_val = len(val)
    
    recons = []
    labels = []
    for imgs, kspace, mask, fimg, target in tqdm(val_loader):
        imgs = imgs.cuda(args.gpu_ind[0])
        kspace = kspace.cuda(args.gpu_ind[0])
        mask = mask.cuda(args.gpu_ind[0]).byte()
        target = target.cuda(args.gpu_ind[0])
        
        with torch.no_grad():            
            if args.task.endswith('class') or args.task.endswith('pred'):
                imgs = imgs.unsqueeze(1)
                
            preds = nn.parallel.data_parallel(task_net, imgs, args.gpu_ind)
            #if args.task.endswith('class'):                
                #preds = preds > 0
        
        for i in range(target.shape[0]):
            label = torch.squeeze(target[i,...]).cpu().detach().numpy()
            if args.task.endswith('recon'):
                pred = torch.squeeze(dc_output[i,...]).cpu().detach().numpy()
            else:
                pred = torch.squeeze(preds[i,...]).cpu().detach().numpy()
                
            recons.append(pred)
            labels.append(label)
        
    return torch.squeeze(preds).cpu().detach().numpy(), torch.squeeze(target).cpu().detach().numpy(), torch.squeeze(imgs).cpu().detach().numpy()

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
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3,help='Learning rate', dest='lr')
    parser.add_argument('-da', '--task', dest='task', type=str, default='Task5_ADNI_ADCN_class',help='dataset') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='2',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=1,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=1,help='Checkpoint saving step')
    parser.add_argument('--acceleration', dest='acceleration', type=float, default=4,help='acceleration')
    parser.add_argument('--center_fraction', dest='center_fraction', type=float, default=0.08,help='center_fraction')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    parser.add_argument('--method', dest='method', type=str, default='',help='method')

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


    dir_name = 'inf_metric'
    os.makedirs(dir_name, exist_ok=True)
    logging.info(f'directory named {dir_name} is made')

    
    if args.task.endswith('seg'):
        from util.dataset_vn_cl_seg import BasicDataset
        from unet import UNet
        metrics = ['IoU','DICE']
        def metr_cal1(x,y):
            return np.nanmean([np.sum((a>0)*(b>0))/np.sum((a>0)+(b>0)>0) for (a,b) in zip(x,y)])
        def metr_cal2(x,y):
            return np.nanmean([2*np.sum((a>0)*(b>0))/(np.sum((a>0))+np.sum((b>0))) for (a,b) in zip(x,y)])

        direc = 'Segmentation'
        task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
        task_net .to(device=device)
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_each_task_chk/{args.task}/task_net_step_best.pth'
        print(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 100
         
    elif args.task.endswith('class'):
        from util.dataset_vn_cl_class import BasicDataset
        from Resnet import Resnet
        metrics = ['Acc','AUC']
        
        def metr_cal1(x,y):
            return np.mean([((a>0)==b) for (a,b) in zip(x,y)])
        def metr_cal2(x,y):
            auc = 0
            for th in range(100):
                auc += np.mean([((a>(np.log(0.01*th/(1-0.01*th))))==b) for (a,b) in zip(x,y)])*0.01
            return auc

        direc = 'Classification'
        if args.task == 'Task4_IXI-T1_Sex_class':
            flattened_shape=[-1, 512, 5, 7, 2]
        elif args.task == 'Task5_ADNI_ADCN_class':
            flattened_shape = [-1, 512, 6, 8, 2]
        task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape,dropout = 0)
        
        #task_net = Resnet(n_classes = args.class_num+1)
        task_net.to(device=device)

        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_each_task_chk/{args.task}/task_net_step_best.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 5
        
    elif args.task.endswith('pred'):
        from util.dataset_vn_cl_pred import BasicDataset
        from Resnet import Resnet
        metrics = ['MAE','RMSE']
        def metr_cal1(x,y):
            return np.mean([np.abs(a-b) for (a,b) in zip(x,y)])

        def metr_cal2(x,y):
            return np.sqrt(np.mean([(a-b)**2 for (a,b) in zip(x,y)]))

        direc = 'Regression'
        flattened_shape = [-1, 512, 6, 7, 2]
        task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape,dropout = 0)
        task_net.to(device=device)
        chck = f'/home/hwihun/clcl/cl_module/checkpoints/Trn_each_task_chk/{args.task}/task_net_step_best.pth'
        print(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 5
        
    else:
        from util.dataset_vn import BasicDataset
        metrics = ['PSNR','SSIM']
        from skimage.metrics import structural_similarity
        from skimage.metrics import peak_signal_noise_ratio
        def metr_cal1(x,y):
            return np.mean([peak_signal_noise_ratio(a,b,data_range = np.max(b)) for (a,b) in zip(x,y)])
        def metr_cal2(x,y):
            return np.mean([structural_similarity(a,b) for (a,b) in zip(x,y)])
        task_net = []
        args.batchsize = 100
        
    val = BasicDataset(args,'test')
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    #task_list = ['Task0_Fastmri_recon','Task1_OASIS1_Tissue_seg','Task2_ATLAS_2_Lesion_seg','Task3_BRATS_Tumor_seg','Task4_IXI-T1_Sex_class','Task5_ADNI_AD_class','Task6_OASIS3_age_pred']

    task_list = ['Task0_Fastmri_recon','Task1_OASIS1_Tissue_seg','Task3_BRATS_Tumor_seg','Task4_IXI-T1_Sex_class','Task5_ADNI_ADCN_class']    #task_ind_start = task_list.index(args.task)
    task_ind_start = 0
    
    method = args.method


    metric1_means = []
    metric2_means = []
    
    args.task_trained = args.task
    recons, labels,img_pred  = predict_net(task_net,args,val_loader)
    met1 = metr_cal1(recons,labels)
    met2 = metr_cal2(recons,labels)
    

    
    print(f'{method}: {metrics[0]}: {met1}, {metrics[1]}: {met2}')
    metric1_means.append(np.mean(met1))
    metric2_means.append(np.mean(met2))
    
    print(f'{method}: {metrics[0]} mean: {np.mean(metric1_means)}, {metrics[1]} mean: {np.mean(metric2_means)}')
        
    np.save(f'inf_metric/each_task_{metrics[0]}_{args.task}_means.npy',metric1_means)
    np.save(f'inf_metric/each_task_{metrics[1]}_{args.task}_means.npy',metric2_means)
    
    path = f'/home/hwihun/clcl/cl_module/inf_img/inf_each_task_{args.task}.pklv4'
    with open(path, 'wb') as f:
        pickle.dump([recons,labels,img_pred], f, protocol=4)