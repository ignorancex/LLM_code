import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torch.utils.data
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
# from sklearn.model_selection import StratifiedGroupKFold

from utils.logger import Logger
from tqdm import tqdm
from restore_module import TinyUNet
from networks.fcn import FCN    # optional for experiments
from moac_dataset import MOACDataset
# from utils.edge_loss import EdgeLoss

# argparse here
parser = argparse.ArgumentParser(description='MOAC-UNET')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=32)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()

### write model configs here
root =  '/kaggle/working/MOAC_deep/'
save_root = './run'
pth_location = '../model_FCN1.pth'
logger = Logger(save_root)
logger.global_step = 0
# n_splits = 5
train_val_percent = 0.8
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# skf = StratifiedGroupKFold(n_splits=n_splits)

trainset = MOACDataset('/kaggle/working/MOAC_deep/dataset', 'train')
testset = MOACDataset('/kaggle/working/MOAC_deep/dataset', 'test')

train_size = int(len(trainset) * train_val_percent)
val_size = len(trainset) - train_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True)
testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = False)

# model = TinyUNet(4,1,bilinear=True)
model = FCN(4,1)
model = model.cuda()

criterion = nn.MSELoss()
# criterion2 = EdgeLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)

lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[7,20],gamma=0.3)
logger.auto_backup('./')

def train(trainloader, model, criterion, optimizer, lrsch, logger, args, epoch):
    model.train()
    losses = 0.
    loss_logger = 0.
    mse_list = []
    logger.update_step()
    for gt, noised, gt_sum in tqdm(trainloader,ascii=True,ncols=60):
        optimizer.zero_grad()
        outs = model(noised.cuda())
        # print("opt tensor:",out)
        gt = gt.cuda()
        gt_sum = gt_sum.cuda()
        loss_batch = 100*criterion(outs,gt) + criterion(torch.sum(outs.squeeze(1),dim=1),gt_sum)
        # loss_batch = 100*criterion(dec,gt)+\
        #     100*criterion(feax,feay)+\
        #     criterion(torch.sum(dec.squeeze(1),dim=1),gt_sum)    # dae only
        loss_batch.backward()
        loss_logger += loss_batch.item()    
        optimizer.step()
        sum_mse = criterion(torch.sum(outs.squeeze(1),dim=1),gt_sum)
        # print(sum_mse)  # debug
        mse_list.append(sum_mse.cpu().detach())
    lrsch.step()
    # wiener_mse = criterion(torch.sum(noised.cuda()[:,2,:,:].squeeze(),dim=1),gt_sum).cpu().detach()   # 取前两个样本
    # net_mse = criterion(torch.sum(outs.squeeze(1),dim=1),gt_sum).cpu().detach()
    # print('Compare with Wiener MSE:',wiener_mse)    # compare with winener mse
    # print('Network MSE:',net_mse)
    loss_logger /= len(trainloader)
    print("Train loss:",loss_logger)
    log_metric('Train', mse_list, logger,loss_logger)
    if not (logger.global_step % args.save_interval):
        logger.save(model,optimizer, lrsch, criterion)
        
def validate(valloader, model, criterion, optimizer, lrsch, logger, args):
    model.eval()
    losses = 0.
    loss_logger = 0.
    mse_list = []

    for gt, noised, gt_sum in tqdm(valloader,ascii=True,ncols=60):
        with torch.no_grad():
            outs = model(noised.cuda())
        gt = gt.cuda()
        gt_sum = gt_sum.cuda()
        # print("label:",label)
        
        loss_batch = 100*criterion(outs,gt) + criterion(torch.sum(outs.squeeze(1),dim=1),gt_sum)
        loss_logger += loss_batch.item()    
        sum_mse = criterion(torch.sum(outs.squeeze(1),dim=1),gt_sum)
        mse_list.append(sum_mse.cpu().detach())
    # wiener_mse = criterion(torch.sum(noised.cuda()[:,2,:,:].squeeze(),dim=1),gt_sum).cpu().detach()   # 取前两个样本
    # net_mse = criterion(torch.sum(outs.squeeze(1),dim=1),gt_sum).cpu().detach()
    # print('Compare with Wiener MSE:',wiener_mse)    # compare with winener mse
    # print('Network MSE:',net_mse)
    loss_logger /= len(valloader)
    print("Val loss:",loss_logger)

    avg_mse = log_metric('Val', mse_list, logger,loss_logger)

    return 10./avg_mse, model.state_dict()  # mse取逆，作为分数（mse越低分数越高）
        
test = validate # reuse function

def log_metric(prefix,mse,logger,loss):
    avg_mse = np.mean(np.array(mse))
    # auc = roc_auc_score(target, prob)
    logger.log_scalar(prefix+'/loss',loss,print=False)
    # logger.log_scalar(prefix+'/AUC',auc,print=True)
    logger.log_scalar(prefix+'/'+'MSE',avg_mse, print= True)

    return avg_mse

if __name__=='__main__':
    score_last = 0
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        train(trainloader, model,criterion,optimizer,lrsch,logger,args,i)
        score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args)
        if score > score_last:
            score_last = score
            print('Save result at epoch {}'.format(i))
            torch.save(model_save, pth_location)
    print('Final testing:')
    score,_ = test(testloader,model,criterion,optimizer,lrsch,logger,args)
