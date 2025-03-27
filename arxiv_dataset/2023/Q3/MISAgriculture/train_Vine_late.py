import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.utils import calculate_accuracy,calculate_result,calculate_accuracy_bin
#from utils.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from tqdm import tqdm

from models.segnet import SegNet
from dataloaders.VINE import MVARGEMDataset
import cv2
import matplotlib.pyplot as plt

import network
import utils_deep
from utils_deep import ext_transforms as et

#from torch.utils.data import DataLoader
#root = "/home/deep/NunoCunha/src/"  
root_val = "/media/deep/datasets/datasets/vineyards/valdoeiro/"
root_esac = "/media/deep/datasets/datasets/vineyards/esac/"
root_qbaixo = "/media/deep/datasets/datasets/vineyards/qbaixo/"

batch_size = 16
epochs = 100
lr_start  = 0.0001 #0.0001
combination = 't1'
model_name='deeplabv3' #segnet or deeplabv3
save_name = 'DeepLab_'+combination+'_LateFusion_'+str(batch_size)
#fusion_type='rgb' #rgb | ndvi | early | late

# config
model_dir = 'weights/'
#model_dir2 = 'weights/SegNet/Masks/'

#lr_decay  = 0.01
#train_losses = []
running_loss = 0
tp, fp, tn, fn = 0, 0, 0, 0
accuracies, f1_scores, recalls, precisions, dice_scores, ious = [], [], [], [], [], []

def train_epoch(epo, model, model_ndvi, train_loader, optimizer, model_name):
    lr_this_epo = lr_start# * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    loss_fn = nn.BCEWithLogitsLoss()
    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    model.train()
    model_ndvi.train()
    for it, data in enumerate(train_loader):
        rgb, ndvi, mask, id = data

        if len(ndvi.shape)<=3:
            ndvi = ndvi.unsqueeze(1) #[32,1,240,240]

        if args.gpu >= 0:
            rgb = rgb.cuda(args.gpu)    #[32,3,240,240]
            ndvi = ndvi.float().cuda(args.gpu)  #[32,240,240]
            mask = mask.cuda(args.gpu)#.requires_grad_(True)  #[32,1,240,240]
        #print(mask.min().item())
        #print(mask.max().item())
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  #[0;1]  #[32,3,240,240]  

        #input = torch.cat([rgb, ndvi], dim=1)

        optimizer.zero_grad()
        
        #output = torch.cat([output_rgb, output_ndvi], dim=1)    #[32, 2, 240, 240]
        #output = output.mean(dim=1, keepdim=True)                #[32, 1, 240, 240]
        if model_name == 'segnet':
            output_rgb, _ = model(rgb)
            output_ndvi, _ = model_ndvi(ndvi)


            
        if model_name == 'deeplabv3':
            output_rgb = model(rgb)
            output_ndvi = model_ndvi(ndvi)

        
        probabilities_rgb = torch.sigmoid(output_rgb)
        #probabilities = output_rgb # torch.softmax(output_rgb,dim=0)
        probabilities_ndvi =  torch.sigmoid(output_ndvi)

        fused = torch.concatenate((probabilities_rgb, probabilities_ndvi),dim=1)
        probabilities = torch.mean(fused, dim=1).unsqueeze(dim=1)

        #print(probabilities[0].min().item())
        #print(probabilities[0].max().item())
        #probabilities = torch.sigmoid(output)
        #predictions = (probabilities > 0.5).float()

        loss = loss_fn(probabilities, mask)

        loss.backward()
        optimizer.step()
       
        #acc = calculate_accuracy_bin(output, mask)
        loss_avg += float(loss)
        #acc_avg  += float(acc) 

        cur_t = time.time()
        #if cur_t-t > 5:
        #    print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.2f%%' \
        #        % (epo, args.epoch_max, it+1, len(train_loader), (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
        #    t += 5

    #content = '| epo:%s/%s lr:%.6f train_loss_avg:%.4f train_acc_avg:%.3f%%' \
            #% (epo, args.epoch_max, lr_this_epo, loss_avg/len(train_loader), (acc_avg/len(train_loader)))
    #print(content)
    #with open(log_file, 'a') as appender:
        #appender.write(content)
        #appender.write('\n')
    return acc_avg

def test_epoch(model, model_ndvi, model_name, test_dataloader):
    loss_avg = 0.
    acc_avg  = 0.
    cf = np.zeros((2, 2))
    
    #model.eval()
    with torch.no_grad():
       
        for it, data in enumerate(test_dataloader):
            rgb, ndvi, mask, id = data   

            if len(ndvi.shape)<=3:
                ndvi = ndvi.unsqueeze(1) #[32,1,240,240]
            if args.gpu >= 0:
                rgb = rgb.cuda(args.gpu)    #[32,3,240,240]
                ndvi = ndvi.float().cuda(args.gpu)  #[32,240,240]
                mask = mask.cuda(args.gpu).requires_grad_(True)  #[32,1,240,240]


            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  #[0;1]   
            #nput = torch.cat([rgb, ndvi], dim=1)
            
            #output,_ = model(rgb)
   
            if model_name == 'segnet':
                output_rgb, _ = model(rgb)
                output_ndvi, _ = model_ndvi(ndvi)
            
            if model_name == 'deeplabv3':
                output_rgb = model(rgb)
                output_ndvi = model_ndvi(ndvi)

            probabilities_rgb = torch.sigmoid(output_rgb)
            probabilities_ndvi = torch.sigmoid(output_ndvi)
            #print(probabilities_rgb[0].min().item())
            #print(probabilities_rgb[0].max().item())
            fused = torch.concatenate((probabilities_rgb, probabilities_ndvi),dim=1)
            probabilities = torch.mean(fused, dim=1).unsqueeze(dim=1)

            #probabilities = torch.mean(torch.stack([probabilities_rgb, probabilities_ndvi]), dim=0)

            #probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).float()

            #acc = calculate_accuracy_bin(output, mask)
            #acc_avg  += float(acc)

            tp = torch.sum((predictions == 1) & (mask == 1)).item()
            fp = torch.sum((predictions == 1) & (mask == 0)).item()
            tn = torch.sum((predictions == 0) & (mask == 0)).item()
            fn = torch.sum((predictions == 0) & (mask == 1)).item()
            acc = (tp + tn) / (tp + fp + tn + fn + 1e-7)

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1_score = 2 * precision * recall / (precision + recall + 1e-7)

            intersection = torch.sum((predictions == 1) & (mask == 1)).item()
            union = torch.sum((predictions == 1) | (mask == 1)).item()
            iou = intersection / (union + 1e-7)

            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1_score)
            ious.append(iou)
            accuracies.append(acc)

            for gtcid in range(2): 
                for pcid in range(2):
                    gt_mask      = mask == gtcid 
                    pred_mask    = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())

    overall_acc, acc, IoU = calculate_result(cf)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1_score = np.mean(f1_scores)
    mean_iou = np.mean(ious)
    mean_accuracy = np.mean(accuracies)

    mean_accuracy_percent = round(mean_accuracy * 100, 2)
    mean_precision_percent = round(mean_precision * 100, 2)
    mean_recall_percent = round(mean_recall * 100, 2)
    mean_f1_score_percent = round(mean_f1_score * 100, 2)
    mean_iou_percent = round(mean_iou * 100, 2)
    
    content =  f"| - test- Acc: {mean_accuracy_percent}%. Acc(class): {acc}%. Prec: {mean_precision_percent}%. Recall: {mean_recall_percent}%. F1: {mean_f1_score_percent}%. IoU: {mean_iou_percent}%.\n"
    #content += f"| Accuracy of each class: {acc}%\n"
    #content += f"| Class accuracy avg:: {acc.mean()}%\n"
    print(content)

    with open(log_file, 'a') as appender:
        appender.write(content)
        appender.write('\n')
        
    return mean_accuracy


def main(log_file):

    torch.manual_seed(0)
    np.random.seed(0)

    num_classes=1
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50',num_classes=1, pretrained=False)
    #model = SegNet(num_classes, n_init_features = rgb_channels)
    #model_ndvi = SegNet(num_classes, n_init_features = 1)
    model = network.modeling.__dict__['deeplabv3_resnet50'](in_channels=3,num_classes=1, output_stride=8)
    model_ndvi = network.modeling.__dict__['deeplabv3_resnet50'](in_channels=1,num_classes=1, output_stride=8)
    #model = eval()
    if args.gpu >= 0: 
        model.cuda(args.gpu)
        model_ndvi.cuda(args.gpu)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=0.0005) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start)
    optimizer_ndvi = torch.optim.AdamW(model_ndvi.parameters(), lr=lr_start)


    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file, map_location={'cuda:0':'cuda:1'}))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))

        model_ndvi.load_state_dict(torch.load(checkpoint_model_ndvi, map_location={'cuda:0':'cuda:1'}))
        optimizer_ndvi.load_state_dict(torch.load(checkpoint_optim_ndvi))
        print('done!')

    train_loader_val    = MVARGEMDataset(root=root_val, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes)
    train_loader_esac   = MVARGEMDataset(root=root_esac, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes)
    train_loader_qbaixo = MVARGEMDataset(root=root_qbaixo, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes) 
    
    #train_loader_val.set_aug_flag(False)
    #train_loader_esac.set_aug_flag(False)
 
    if combination == 't1':
        train_dataloader_combined = torch.utils.data.ConcatDataset([train_loader_val, train_loader_esac])
        train_dataloader_validation = DataLoader(train_loader_qbaixo, batch_size=batch_size, shuffle=False)

    elif combination == 't2':
        train_dataloader_combined = torch.utils.data.ConcatDataset([train_loader_val, train_loader_qbaixo])
        train_dataloader_validation = DataLoader(train_loader_esac, batch_size=batch_size, shuffle=False)

    elif combination == 't3':
        train_dataloader_combined = torch.utils.data.ConcatDataset([train_loader_qbaixo, train_loader_esac])
        train_dataloader_validation = DataLoader(train_loader_val, batch_size=batch_size, shuffle=False)

    # Create Training dataset
    train_dataloader_combined = DataLoader(train_dataloader_combined, batch_size=batch_size, shuffle=True)

    if os.path.exists(final_model_file):
        os.remove(final_model_file)

    if os.path.exists(final_model_ndvi):
        os.remove(final_model_ndvi)

    best_f1 = 0.0
    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        
        #print('\n| epo #%s begin...' % epo)
        
        acc_avg = train_epoch(epo, model, model_ndvi,train_dataloader_combined, optimizer, model_name)
        #validation(epo, model, val_loader)
        f1_test = test_epoch(model, model_ndvi, model_name, train_dataloader_validation)  


        # save check point model
        if f1_test > best_f1:
            best_f1 = f1_test
            torch.save(model.state_dict(), best_model_file)
            torch.save(model_ndvi.state_dict(), best_ndvi)

        
        #print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)

        torch.save(model_ndvi.state_dict(), checkpoint_model_ndvi)
        torch.save(optimizer_ndvi.state_dict(), checkpoint_optim_ndvi)
        #print('done!')

        
    # Rename the checkpoint model file to the final model file
    os.rename(checkpoint_model_file, final_model_file)
    os.rename(checkpoint_model_ndvi, final_model_ndvi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segnet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default=save_name)
    parser.add_argument('--batch_size',  '-B',  type=int, default=batch_size)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=epochs)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_model_file = os.path.join(model_dir, 'tmp.pth')
    checkpoint_model_ndvi = os.path.join(model_dir, 'tmp_ndvi.pth')
    checkpoint_optim_file = os.path.join(model_dir, 'tmp.optim')
    checkpoint_optim_ndvi = os.path.join(model_dir, 'tmp_ndvi.optim')
    best_model_file       = os.path.join(model_dir, 'best.pth')  
    best_ndvi             = os.path.join(model_dir, 'best_ndvi.pth')  
    final_model_file      = os.path.join(model_dir, 'final.pth')
    final_model_ndvi      = os.path.join(model_dir, 'final_ndvi.pth')
    log_file              = os.path.join(model_dir, 'log.txt')

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)
    
    main(log_file)