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

from models.segnet0 import SegNet
from dataloaders.VARGEM import MVARGEMDataset
#from torch.utils.data import DataLoader
root = "/home/deep/NunoCunha/src/"  # path to the root directory of the dataset 
batch_size = 32
epochs = 100
rgb_channels = 3
lr_start  = 0.001 #6

# config
model_dir = 'weights/'
#model_dir2 = 'weights/SegNet/Masks/'

#lr_decay  = 0.01
#train_losses = []
running_loss = 0
tp, fp, tn, fn = 0, 0, 0, 0
accuracies, f1_scores, recalls, precisions, dice_scores, ious = [], [], [], [], [], []

def train(epo, model,model_ndvi, train_loader, optimizer):
    lr_this_epo = lr_start# * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    #train_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    model.train()

    for it, data in enumerate(train_loader):
        rgb,nir,red,mask,id = data

        if args.gpu >= 0:
            rgb = rgb.cuda(args.gpu)
            nir = nir.cuda(args.gpu)
            red = red.cuda(args.gpu)
            mask = mask.cuda(args.gpu)

        ndvi = (nir - red) / (nir + red)
        min_value = torch.min(ndvi)
        max_value = torch.max(ndvi)   
        ndvi = (ndvi - min_value) / (max_value - min_value)

        
        optimizer.zero_grad()

        output_rgb, _ = model(rgb)
        output_ndvi, _ = model_ndvi(ndvi)
       
        probabilities_rgb = torch.sigmoid(output_rgb)

        probabilities_ndvi =  torch.sigmoid(output_ndvi)

        fused = torch.concatenate((probabilities_rgb, probabilities_ndvi),dim=1)
        probabilities = torch.mean(fused, dim=1).unsqueeze(dim=1)
        loss = loss_fn(probabilities, mask)


        loss.backward()
        optimizer.step()
        
        #probabilities = torch.sigmoid(output)
        #predictions = (probabilities > 0.5).float()
        
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

def test(model, model_ndvi,test_dataloader):
    loss_avg = 0.
    acc_avg  = 0.
    cf = np.zeros((2, 2))
    with torch.no_grad():
        for it, data in enumerate(test_dataloader):
            rgb,nir,red,mask,id = data

            if args.gpu >= 0:
                rgb = rgb.cuda(args.gpu)
                nir = nir.cuda(args.gpu)
                red = red.cuda(args.gpu)
                mask = mask.cuda(args.gpu)

            ndvi = (nir - red) / (nir + red)
            min_value = torch.min(ndvi)
            max_value = torch.max(ndvi)   
            ndvi = (ndvi - min_value) / (max_value - min_value)


            output_rgb, _ = model(rgb)
            output_ndvi, _ = model_ndvi(ndvi)
       
            probabilities_rgb = torch.sigmoid(output_rgb)
            probabilities_ndvi =  torch.sigmoid(output_ndvi)

            fused = torch.concatenate((probabilities_rgb, probabilities_ndvi),dim=1)
            probabilities = torch.mean(fused, dim=1).unsqueeze(dim=1)

            predictions = (probabilities > 0.5).float()

          # Calculate the pixel-wise precision, recall, F1 score, and Dice score
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
    model = SegNet(num_classes, n_init_features = rgb_channels)
    model_ndvi = SegNet(num_classes, n_init_features = rgb_channels)
    #model = eval()
    if args.gpu >= 0: 
        model.cuda(args.gpu)
        model_ndvi.cuda(args.gpu)

    #optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=0.0005) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start)
    optimizer_ndvi = torch.optim.AdamW(model.parameters(), lr=lr_start)


    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file, map_location={'cuda:0':'cuda:1'}))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))

        model_ndvi.load_state_dict(torch.load(checkpoint_model_ndvi, map_location={'cuda:0':'cuda:1'}))
        optimizer_ndvi.load_state_dict(torch.load(checkpoint_optim_ndvi))
        print('done!')

    #train_loader = MVARGEMDataset( root=root, set='train', rgb_dir = 'rgb', mask_dir = 'GT_color', dsm_dir = 'nir', num_classes = num_classes)
    train_loader = MVARGEMDataset(root=root, set='train', rgb_dir = 'RGB', mask_dir = 'Masks', nir_dir = 'NIR', red_dir = 'RED', num_classes = num_classes)
    #train_loader.set_aug_flag(True)
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

    test_loader = MVARGEMDataset( root=root, set='test', rgb_dir = 'RGB', mask_dir = 'Masks', nir_dir = 'NIR', red_dir = 'RED', num_classes = num_classes)
    test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)

    if os.path.exists(final_model_file):
        os.remove(final_model_file)

    if os.path.exists(final_model_ndvi):
        os.remove(final_model_ndvi)    

    best_acc = 0.0
    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        
        #print('\n| epo #%s begin...' % epo)
        acc_avg = train(epo, model, model_ndvi,train_dataloader, optimizer)
        #validation(epo, model, val_loader)
        acc_test = test(model,model_ndvi, test_dataloader)


        # save check point model
        if acc_test > best_acc:
            best_acc = acc_test
            torch.save(model.state_dict(), best_model_file)
            torch.save(model_ndvi.state_dict(), best_ndvi)
        
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        torch.save(model_ndvi.state_dict(), checkpoint_model_ndvi)
        torch.save(optimizer_ndvi.state_dict(), checkpoint_optim_ndvi)


        
    # Rename the checkpoint model file to the final model file
    os.rename(checkpoint_model_file, final_model_file)
    os.rename(checkpoint_model_ndvi, final_model_ndvi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segnet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='SegNet')
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