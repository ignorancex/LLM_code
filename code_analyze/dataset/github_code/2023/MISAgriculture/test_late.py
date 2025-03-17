import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import time
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.utils import calculate_accuracy, calculate_result, calculate_accuracy_bin
from train import  model_dir

#---------------- MODEL ----------------------
from models.segnet0 import SegNet
rgb_channels = 3

#---------------- DATASET --------------------
from dataloaders.VARGEM import MVARGEMDataset

from torch.utils.data import DataLoader
root = "/home/deep/NunoCunha/src/"  # path to the root directory of the dataset 
batch_size = 64

def main():
    num_classes=1
    model = SegNet(num_classes, n_init_features = rgb_channels)
    model_ndvi = SegNet(num_classes, n_init_features = rgb_channels)
    
                
    cf = np.zeros((2,2))

    #model = eval(args.model_name)(num_classes=num_classes)

    if args.gpu >= 0: 
          model.cuda(args.gpu)
          model_ndvi.cuda(args.gpu)

    print('| loading model file %s... ' % final_model_file, end='')
    #map_location=torch.device('cuda:0')
    #map_location=torch.device('cpu')
    #checkpoint = torch.load(final_model_file,map_location=torch.device('cpu'))
    model.load_state_dict(torch.load(final_model_file, map_location=torch.device('cuda:0')))
    model_ndvi.load_state_dict(torch.load(best_ndvi, map_location=torch.device('cuda:0')))

    print('done!')

    test_loader = MVARGEMDataset( root=root, set='test', rgb_dir = 'RGB', mask_dir = 'Masks', nir_dir = 'NIR', red_dir = 'RED', num_classes = num_classes)

    test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)
    
    loss_avg = 0.
    acc_avg  = 0.
    tp, fp, tn, fn = 0, 0, 0, 0
    accuracies, f1_scores, recalls, precisions, dice_scores, ious = [], [], [], [], [], []
    model.eval()
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



            for gtcid in range(num_classes): 
                            for pcid in range(num_classes):
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

    content =  f"\n| Pixel Accuracy: {mean_accuracy_percent}%\n"
    #content += f"| Accuracy of each class: {acc}%\n"
    #content += f"| Class accuracy avg:: {acc.mean()}%\n"
    content += f"| Precision: {mean_precision_percent}%\n"
    content += f"| Recall: {mean_recall_percent}%\n"
    content += f"| F1 Score: {mean_f1_score_percent}%\n"
    content += f"| IoU Score: {mean_iou_percent}%\n"
    print(content)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test SegNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='SegNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=batch_size)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir        = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'best.pth')
    best_ndvi        = os.path.join(model_dir, 'best_ndvi.pth')

    assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()