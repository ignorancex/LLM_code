import os
import argparse
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim import *
import torch.nn.functional as F

from sklearn.metrics import *
from sklearn.model_selection import KFold

import sys
sys.path.append('.')
sys.path.append('..')

from src.modules_fea import *
from src.chexpert_dataset import *
from src import logger
from typing import NamedTuple

from fairlearn.metrics import *
from typing import List

class Dataset_Info(NamedTuple):
    beta: float = 0.9999
    gamma: float = 2.0
    samples_per_attr: List[int] = [0,0,0]
    loss_type: str = "sigmoid"
    no_of_classes: int = 2
    no_of_attr: int = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--batch-size', default=50, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--pretrained-weights', default='', type=str)

parser.add_argument('--result_dir', default='../results_feas/chexpert_race_efficientnet_fea_lr1e-4', type=str)
parser.add_argument('--data_dir', default='Data/Chexpert/', type=str)
parser.add_argument('--model_type', default='efficientnet_fea', type=str)
parser.add_argument('--task', default='cls', type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--modality_types', default='slo_fundus', type=str, help='oct_bscans_3d|slo_fundus')
parser.add_argument('--fuse_coef', default=1.0, type=float)
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--attribute_type', default='race', type=str, help='race|gender|hispanic')
parser.add_argument('--subset_name', default='test', type=str)
parser.add_argument("--need_balance", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Oversampling or not")
parser.add_argument('--dataset_proportion', default=1., type=float)
parser.add_argument('--num_classes', default=2, type=int)

# parser.add_argument('--optimizer', type=str, required=True,
#                     choices=['sgd', 'adam', 'adamw', 'adagrad', 'rmsprop', 'adadelta', 'adamax', 'asgd'],
#                     help='Name of the optimizer (e.g., sgd, adam)')
# parser.add_argument('--optimizer_arguments', type=str, default='{}',
#                     help='Additional optimizer parameters as a dictionary (e.g., \'{"weight_decay": 0.01}\')')
# parser.add_argument('--scheduler', type=str, required=True,
#                     choices=['step_lr', 'multi_step_lr', 'exponential_lr', 'cosine_annealing_lr',
#                              'reduce_lr_on_plateau', 'cyclic_lr', 'one_cycle_lr'],
#                     help='Name of the learning rate scheduler')
# parser.add_argument('--scheduler_arguments', type=str, default='{}',
#                     help='Additional scheduler parameters as a dictionary (e.g., \'{"step_size": 30, "gamma": 0.1}\')')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, dataset_info=None, num_classes=2):
    global device

    model.train()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []
    
    preds = []
    gts = []
    attrs = []
    datadirs = []

    feas = []

    preds_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]
    t1 = time.time()
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            input = input.to(device)
            target = target.to(device)
            
            pred, global_fea = model(input)
            # print(pred.shape, target.shape)
            # print(pred.dtype, target.dtype)
            target = target.float()
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
                loss = criterion(pred, target)
                pred_prob = torch.sigmoid(pred.detach())
            elif pred.shape[1] > 1:
                loss = criterion(pred, target.long())
                pred_prob = F.softmax(pred.detach(), dim=1)

            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            feas.append(global_fea.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())

        loss_batch.append(loss.item())
        
        if num_classes == 2:
            top1_accuracy = accuracy(pred_prob.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
        elif num_classes > 2:
            top1_accuracy = accuracy(pred_prob, target, topk=(1,))
        
        top1_accuracy_batch.append(top1_accuracy)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)
    feas = np.concatenate(feas, axis=0)
    
    cur_auc = compute_auc(preds, gts, num_classes=num_classes)
    # acc = np.mean(top1_accuracy_batch)
    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    elif num_classes > 2:
        acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))

    torch.cuda.synchronize()
    t2 = time.time()

    print(f"train ====> epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f} time: {t2 - t1:.4f}")

    t1 = time.time()

    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs, feas
    

def validation(model, criterion, optimizer, validation_dataset_loader, epoch, result_dir=None, dataset_info=None, num_classes=2):
    global device

    model.eval()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []

    preds = []
    gts = []
    attrs = []
    datadirs = []

    feas = []

    preds_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]

    with torch.no_grad():
        for i, (input, target, attr) in enumerate(validation_dataset_loader):
            input = input.to(device)
            target = target.to(device)
            
            pred, global_fea = model(input)
            
            target = target.float()
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
                loss = criterion(pred, target)
                pred_prob = torch.sigmoid(pred.detach())
            elif pred.shape[1] > 1:
                loss = criterion(pred, target.long()).mean()
                pred_prob = F.softmax(pred.detach(), dim=1)

            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())

            feas.append(global_fea.detach().cpu().numpy())

            loss_batch.append(loss.item())

            if num_classes == 2:
                top1_accuracy = accuracy(pred_prob.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
            elif num_classes > 2:
                top1_accuracy = accuracy(pred_prob, target, topk=(1,))
        
            top1_accuracy_batch.append(top1_accuracy)
        
    loss = np.mean(loss_batch)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)

    feas = np.concatenate(feas, axis=0)
    
    cur_auc = compute_auc(preds, gts, num_classes=num_classes)

    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    elif num_classes > 2:
        acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))

    print(f"test <==== epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f}")

    return loss, acc, cur_auc, preds, gts, attrs, feas


args = parser.parse_args([])
print("BATCHSIZE", args.batch_size)
if args.seed < 0:
    args.seed = int(np.random.randint(10000, size=1)[0])
set_random_seed(args.seed)

logger.log(f'===> random seed: {args.seed}')

logger.configure(dir=args.result_dir, log_suffix='train')

if args.model_type == 'vit' or args.model_type == 'swin':
    args.image_size = 224
    
with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# file_path = "../Data/Chexpert/", , list_file = "chexpert_sample_" + split + ".csv",  )

trn_havo_dataset = CheXpertDataset(file_path = args.data_dir, split = 'train', disease = 'Pleural Effusion', attribute = 'gender')
tst_havo_dataset = CheXpertDataset(file_path = args.data_dir, split = 'test', disease = 'Pleural Effusion', attribute = 'gender')

args.num_classes = 2
logger.log(f'there are {args.num_classes} classes in total')
# logger.log(trn_havo_dataset.disease_mapping)

logger.log(f'trn patients {len(trn_havo_dataset)} with {len(trn_havo_dataset)} samples, val patients {len(tst_havo_dataset)} with {len(tst_havo_dataset)} samples')

train_dataset_loader = torch.utils.data.DataLoader(
    trn_havo_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)

validation_dataset_loader = torch.utils.data.DataLoader(
    tst_havo_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)

# samples_per_attr = get_num_by_group_(train_dataset_loader)
# logger.log(f'group information:')
# logger.log(samples_per_attr)
ds_info = Dataset_Info(no_of_attr=2)

best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
lastep_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'last_{args.perf_file}')

acc_head_str = ''
auc_head_str = ''
dpd_head_str = ''
eod_head_str = ''
esacc_head_str = ''
esauc_head_str = ''
group_disparity_head_str = ''
if args.perf_file != '':
    if not os.path.exists(best_global_perf_file):
        for i in range(len(samples_per_attr)):
            auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(len(samples_per_attr[i]))]) + ', '
        dpd_head_str += ', '.join([f'dpd_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        eod_head_str += ', '.join([f'eod_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        esacc_head_str += ', '.join([f'esacc_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        esauc_head_str += ', '.join([f'esauc_attr{i}' for i in range(len(samples_per_attr))]) + ', '

        group_disparity_head_str += ', '.join([f'std_group_disparity_attr{i}, max_group_disparity_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        
        with open(best_global_perf_file, 'w') as f:
            f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

in_dim = 3
out_dim = 1
criterion = nn.BCEWithLogitsLoss()
model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
model = model.to(device)

scaler = torch.cuda.amp.GradScaler()

optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
#
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# optimizer=get_optimizer(args.optimizer,model.parameters(),args.lr,args.optimizer_arguments)
# scheduler=get_scheduler(args.scheduler,optimizer,args.scheduler_arguments)

# optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.lr, **eval(args.optimizer_arguments))
# scheduler=get_scheduler(args.scheduler,optimizer,**eval(args.scheduler_arguments))

start_epoch = 0
best_top1_accuracy = 0.

if args.pretrained_weights != "":
    checkpoint = torch.load(args.pretrained_weights)

    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

total_iteration = len(trn_havo_dataset)//args.batch_size

best_auc_groups = None
best_acc_groups = None
best_pred_gt_by_attr = None
best_auc = sys.float_info.min
best_acc = sys.float_info.min
best_es_acc = sys.float_info.min
best_es_auc = sys.float_info.min
best_ep = 0
best_between_group_disparity = None

for epoch in range(start_epoch, args.epochs):
    train_loss, train_acc, train_auc, trn_preds, trn_gts, trn_attrs, train_feas = train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, dataset_info=ds_info, num_classes=args.num_classes)
    test_loss, test_acc, test_auc, tst_preds, tst_gts, tst_attrs, test_feas = validation(model, criterion, optimizer, validation_dataset_loader, epoch, dataset_info=ds_info, num_classes=args.num_classes)

    np.savez(args.result_dir+'/Features_'+str(epoch)+'.npz', feas=test_feas, attrs = tst_attrs, labels = tst_gts)

    
    scheduler.step()

    val_es_acc, val_es_auc, val_aucs_by_attrs, val_dpds, val_eods, between_group_disparity = evalute_comprehensive_perf(tst_preds, tst_gts, tst_attrs, num_classes=args.num_classes)

    if best_auc <= test_auc:
        best_auc = test_auc
        best_acc = test_acc
        best_ep = epoch
        best_auc_groups = val_aucs_by_attrs
        best_dpd_groups = val_dpds
        best_eod_groups = val_eods
        best_es_acc = val_es_acc
        best_es_auc = val_es_auc
        best_between_group_disparity = between_group_disparity

        state = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'train_auc': train_auc,
        'test_auc': test_auc
        }
        torch.save(state, os.path.join(args.result_dir, f"model_ep{epoch:03d}.pth"))

    logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
    logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
    logger.log(best_auc_groups)
    
    # if args.result_dir is not None:
    #     np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'), 
    #                 val_pred=tst_preds, val_gt=tst_gts, val_attr=tst_attrs)

    logger.logkv('epoch', epoch)
    logger.logkv('trn_loss', round(train_loss,4))
    logger.logkv('trn_acc', round(train_acc,4))
    logger.logkv('trn_auc', round(train_auc,4))

    logger.logkv('val_loss', round(test_loss,4))
    logger.logkv('val_acc', round(test_acc,4))
    logger.logkv('val_auc', round(test_auc,4))

    for ii in range(len(val_es_acc)):
        logger.logkv(f'val_es_acc_attr{ii}', round(val_es_acc[ii],4))
    for ii in range(len(val_es_auc)):
        logger.logkv(f'val_es_auc_attr{ii}', round(val_es_auc[ii],4))
    for ii in range(len(val_aucs_by_attrs)):
        for iii in range(len(val_aucs_by_attrs[ii])):
            logger.logkv(f'val_auc_attr{ii}_group{iii}', round(val_aucs_by_attrs[ii][iii],4))

    for ii in range(len(between_group_disparity)):
        logger.logkv(f'val_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0],4))
        logger.logkv(f'val_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1],4))

    for ii in range(len(val_dpds)):
        logger.logkv(f'val_dpd_attr{ii}', round(val_dpds[ii],4))
    for ii in range(len(val_eods)):
        logger.logkv(f'val_eod_attr{ii}', round(val_eods[ii],4))

    logger.dumpkvs()

if args.perf_file != '':
    if os.path.exists(best_global_perf_file):
        with open(best_global_perf_file, 'a') as f:

            esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
            esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

            auc_head_str = ''
            for i in range(len(best_auc_groups)):
                auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

            group_disparity_str = ''
            for i in range(len(best_between_group_disparity)):
                group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
            
            dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
            eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

            path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}'
            f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')
            
os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}')
