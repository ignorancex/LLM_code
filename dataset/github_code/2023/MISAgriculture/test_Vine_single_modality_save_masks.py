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
import network

from utils.utils import calculate_accuracy, calculate_result, calculate_accuracy_bin
from train_Vine_deeplab import  model_dir


#---------------- MODEL ----------------------
from models.segnet0 import SegNet

#---------------- DATASET --------------------



from dataloaders.VINE import MVARGEMDataset as VIN_MVARGEMDataset
from dataloaders.VARGEM import MVARGEMDataset as MAIZE_MVARGEMDataset

from torch.utils.data import DataLoader
#root = "/home/deep/NunoCunha/src/"  
root_val = "/media/deep/datasets/datasets/vineyards/valdoeiro/"
root_esac = "/media/deep/datasets/datasets/vineyards/esac/"
root_qbaixo = "/media/deep/datasets/datasets/vineyards/qbaixo/"

root_vg = "/home/deep/NunoCunha/src/"  # path to the root directory of the VG dataset 


def compute_performance_metrics(predictions,mask):
    # Compute Metrics 
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

    return{'acc':acc,'f1':f1_score,'iou':iou,'precision':precision,'recall':recall}



def select_testset(dataset,batch_size=32):
    num_classes = 1

    assert dataset in ['val','esac','qbaixo','vg'],'Dataset not recognized'

    if dataset.lower() == 'val':
        loader = VIN_MVARGEMDataset(root=root_val, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes) 
    elif dataset.lower() == 'esac':
        loader = VIN_MVARGEMDataset(root=root_esac, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes) 
    elif dataset.lower() == 'qbaixo':
        loader = VIN_MVARGEMDataset(root=root_qbaixo, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes) 
    elif dataset.lower() == 'vg':
        loader = MAIZE_MVARGEMDataset( root=root_vg, set='test', rgb_dir = 'RGB', mask_dir = 'Masks', nir_dir = 'NIR', red_dir = 'RED', num_classes = num_classes)
    return DataLoader(loader, batch_size=batch_size, shuffle=False)




def select_seg_model(name,modality="RGB"):
    num_classes =1 # Hard coded because all datasets have only one class
    model = None   

    assert name.lower() in ['deeplab','segnet'], "Model not recognized"
    assert modality.lower() in ['rgb','ndvi'],"Modality not recognized"
    # map from madality label to input channels
    if modality.lower()  == "rgb":
        input_channels = 3
    elif  modality.lower()  == "ndvi":
        input_channels = 1
    # Selection of the model
    if name.lower() in 'deeplab':
        model = network.modeling.__dict__['deeplabv3_resnet50'](in_channels=input_channels,num_classes=1, output_stride=8)
    elif name.lower() == 'segnet':
        model = SegNet(num_classes, n_init_features = input_channels)
    return model


def late_fusion(model1, model2, data):
    # Set both models to evaluation mode
    model1.eval()
    model2.eval()

    # Disable gradient computation for faster inference
    with torch.no_grad():
        # Get predictions from the first model
        predictions1 = model1(data)

        # Get predictions from the second model
        predictions2 = model2(data)

        # Perform weighted averaging to combine predictions
        fused_predictions = (weight_model1 * predictions1) + (weight_model2 * predictions2)

    return fused_predictions


def main(checkpoint_model_file):

    
    # GET MODALITY FROM THE WEIGHTS TO BE LOADED
    modality = None
    if 'rgb' in checkpoint_model_file.lower():
        modality = 'rgb'
    elif 'ndvi' in checkpoint_model_file.lower():
        modality = 'ndvi'
    assert modality in ['rgb','ndvi']
  
    # GET DATASET FROM THE WEIGHTS TO BE LOADED
    dataset_name = None
    # T1: valdoeiro + esac 	  | qbaixo
    # T2: valdoeiro + qbaixo  | esac
    # T3: esac +  qbaixo	  | valdoeiro
    if 'vg' in  checkpoint_model_file.lower():
        dataset_name = 'vg'
    elif 't1' in checkpoint_model_file.lower():
        dataset_name = 'qbaixo'
    elif 't3' in checkpoint_model_file.lower():
        dataset_name = 'val'
    elif 't2' in checkpoint_model_file.lower():
        dataset_name = 'esac'
    assert dataset_name in ['vg','qbaixo','val','esac']

    # GET MODEL FROM THE WEIGHTS TO BE LOADED
    model_name = None
    if "segnet" in checkpoint_model_file.lower():
        model_name = 'segnet'
    elif "deeplab" in checkpoint_model_file.lower():
        model_name = 'deeplab'
    assert model_name != None

    # SELCTION OF THE DATASET
    test_dataloader = select_testset(dataset_name,batch_size=32)
    print(f"| Loaded {dataset_name} ... ")
    # SELECTION OF THE MODEL
    model = select_seg_model(model_name,modality)
    model.load_state_dict(torch.load(checkpoint_model_file,map_location=torch.device('cuda:0')))
    print('| Loaded  %s ' % checkpoint_model_file, end='')

    if args.gpu >= 0: 
        model.cuda(args.gpu)

    accuracies, f1_scores, recalls, precisions, ious = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for it, data in enumerate(test_dataloader):
            rgb, ndvi, mask, id = data   

            if len(ndvi.shape)<=3:
               ndvi = ndvi.unsqueeze(1) #[32,1,240,240]

            if args.gpu >= 0:
                rgb = rgb.cuda(args.gpu)    #[32,3,240,240]
                ndvi = ndvi.float().cuda(args.gpu)  #[32,240,240]
                mask = mask.cuda(args.gpu)  #[32,1,240,240]

            # Normalize RGB
            # rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  #[0;1]  #[32,3,240,240]


            if modality == 'rgb':
               input = rgb
            if modality == 'ndvi':
                input = ndvi

            # compute mask
            output, _ = model(input)
            # Get confidence score 
            probabilities = torch.sigmoid(output)

            # Get hard class 
            predictions = (probabilities > 0.5).float()

            metrics = compute_performance_metrics(predictions,mask)
 
            recalls.append(metrics['recall'])
            precisions.append(metrics['precision'])
            f1_scores.append(metrics['f1'])
            ious.append(metrics['iou'])
            accuracies.append(metrics['acc'])


    #overall_acc, acc, IoU = calculate_result(cf)
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
    parser.add_argument('--batch_size',  '-B',  type=int, default=32)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir        = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'best.pth')
    best_ndvi        = os.path.join(model_dir, 'best_ndvi.pth')

    #assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))


    #path_trained_weights = "/home/deep/Mario/multiclass_segmentation/weights/segnet/SegNet_VG_Ndvi/best.pth"
    path_trained_weights = "/home/deep/Mario/multiclass_segmentation/weights/deeplabv3/DeepLab_t2_RGB/best.pth"
    path_trained_weights = "/home/deep/Mario/multiclass_segmentation/weights/DeepLab_t1_RGB/best.pth"
    #path_trained_weights = "/home/deep/Mario/multiclass_segmentation/weights/deeplabv3/DeepLab_t3_RGB/best.pth"

    #path_trained_weights = "/home/deep/Mario/multiclass_segmentation/weights/deeplabv3/DeepLab_t3_NDVI/best.pth"
    #path_trained_weights = "/home/deep/Mario/multiclass_segmentation/weights/deeplabv3/DeepLab_t2_NDVI/best.pth"
    #path_trained_weights = "/home/deep/Mario/multiclass_segmentation/weights/deeplabv3/DeepLab_t1_NDVI/best.pth"
    main(path_trained_weights)