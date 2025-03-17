import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import time
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn.functional as F
from torch.autograd import Variable
#from torch.utils.data import DataLoader
import torch.nn as nn

from utils.utils import calculate_accuracy,calculate_result,calculate_accuracy_bin
#from utils.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from tqdm import tqdm

from models.segnet import SegNet
from dataloaders.VINE import MVARGEMDataset
from dataloaders.VINE import MVARGEMDataset as VIN_MVARGEMDataset
from dataloaders.VARGEM import MVARGEMDataset as MAIZE_MVARGEMDataset

import cv2
import matplotlib.pyplot as plt

import network
import utils_deep
from utils_deep import ext_transforms as et
import pickle


#from torch.utils.data import DataLoader
#root = "/home/deep/NunoCunha/src/"  
root_val = "/media/deep/datasets/datasets/vineyards/valdoeiro/"
root_esac = "/media/deep/datasets/datasets/vineyards/esac/"
root_qbaixo = "/media/deep/datasets/datasets/vineyards/qbaixo/"
root_vg = "/home/deep/NunoCunha/src/"  # path to the root directory of the VG dataset 

batch_size = 32
epochs = 100
lr_start = 0.001
#fusion_type='rgb' #rgb | ndvi | early | late

overal_perfm = "global_perfm.txt"

    
# config
model_dir = 'weights/'
#model_dir2 = 'weights/SegNet/Masks/'

#lr_decay  = 0.01
#train_losses = []
running_loss = 0
tp, fp, tn, fn = 0, 0, 0, 0


def compute_performance_metrics(predictions,mask):

    tp = torch.sum((predictions == 1) & (mask == 1)).item()
    fp = torch.sum((predictions == 1) & (mask == 0)).item()
    tn = torch.sum((predictions == 0) & (mask == 0)).item()
    fn = torch.sum((predictions == 0) & (mask == 1)).item()
    
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-7)

    intersection = torch.sum((predictions == 1) & (mask == 1)).item()
    union = torch.sum((predictions == 1) | (mask == 1)).item()
    iou = intersection / (union + 1e-7)

    return{'acc':acc,'f1':f1_score,'iou':iou,'precision':precision,'recall':recall}



def train_epoch(epo, model, train_loader, optimizer,scheduler,fusion_type):
    lr_this_epo = lr_start# * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    loss_fn = nn.BCEWithLogitsLoss()
    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    model.train()

    for it, data in enumerate(train_loader):
        rgb, ndvi, mask, id = data

        if len(ndvi.shape)<=3:
            ndvi = ndvi.unsqueeze(1) #[32,1,240,240]

        if args.gpu >= 0:
            rgb = rgb.cuda(args.gpu)    #[32,3,240,240]
            ndvi = ndvi.float().cuda(args.gpu)  #[32,240,240]
            mask = mask.cuda(args.gpu) #[32,1,240,240]

        #rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  #[0;1]  #[32,3,240,240]  

        #input = torch.cat([rgb, ndvi], dim=1)

        optimizer.zero_grad()

        if fusion_type == 'NDVI':
            #print(ndvi.shape)
            output,_ = model(ndvi)
        elif fusion_type == 'RGB':
            output,_ = model(rgb)
        elif fusion_type == 'EarlyFusion':
            image_fusion = torch.cat([rgb,ndvi], dim=1) 
            output,_ = model(image_fusion)

        predictions = torch.sigmoid(output)
            
        loss = loss_fn(predictions, mask)    
        loss.backward()
        optimizer.step()
        scheduler.step()
        #acc = calculate_accuracy_bin(output, mask)
        loss_avg += float(loss)
        #acc_avg  += float(acc) 

        cur_t = time.time()

    return acc_avg



def test_epoch(model, test_dataloader,fusion_type):
    loss_avg = 0.
    acc_avg  = 0.
    cf = np.zeros((2, 2))
    
    accuracies, f1_scores, recalls, precisions, dice_scores, ious = [], [], [], [], [], []
    #model.eval()
    with torch.no_grad():
       
        for it, data in enumerate(test_dataloader):
            rgb, ndvi, mask, id = data   
            if len(ndvi.shape)<=3:
                ndvi = ndvi.unsqueeze(1) #[32,1,240,240]
            if args.gpu >= 0:
                rgb = rgb.cuda(args.gpu)    #[32,3,240,240]
                ndvi = ndvi.float().cuda(args.gpu)  #[32,240,240]
                mask = mask.cuda(args.gpu)  #[32,1,240,240]

            #rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  #[0;1]   
            #nput = torch.cat([rgb, ndvi], dim=1)
            
            if fusion_type == 'NDVI':
                #print("test")
                #print(ndvi.shape)
                output,_ = model(ndvi)
            elif fusion_type == 'RGB':
                output,_ = model(rgb)
            elif fusion_type == 'EarlyFusion':
                image_fusion = torch.cat([rgb,ndvi], dim=1) 
                output,_ = model(image_fusion)

            probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).float()
            metricts = compute_performance_metrics(predictions,mask)


            recalls.append(metricts['recall'])
            precisions.append(metricts['precision'])
            f1_scores.append(metricts['f1'])
            ious.append(metricts['iou'])
            accuracies.append(metricts['acc'])


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
    
    content =  f"| - test- Acc: {mean_accuracy_percent}%. Prec: {mean_precision_percent}%. Recall: {mean_recall_percent}%. F1: {mean_f1_score_percent}%. IoU: {mean_iou_percent}%.\n"
    #content += f"| Accuracy of each class: {acc}%\n"
    #content += f"| Class accuracy avg:: {acc.mean()}%\n"
    print(content)

    with open(log_file, 'a') as appender:
        appender.write(content)
        appender.write('\n')
        
    return mean_iou_percent



def save_pred_mask(model,testloader,save_pred_mask_dir,parameters):
    accuracies, f1_scores, recalls, precisions, dice_scores, ious = [], [], [], [], [], []
    #model.eval()
    fusion_type = parameters['fusion_type'] 
    time_duration  = []
    print(len(testloader))
    with torch.no_grad():
        for it, data in enumerate(testloader):
            rgb, ndvi, mask, id = data   
            if len(ndvi.shape)<=3:
                ndvi = ndvi.unsqueeze(1) #[32,1,240,240]
            if args.gpu >= 0:
                rgb = rgb.cuda(args.gpu)    #[32,3,240,240]
                ndvi = ndvi.float().cuda(args.gpu)  #[32,240,240]
                mask = mask.cuda(args.gpu)  #[32,1,240,240]

            #rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  #[0;1]   
            #nput = torch.cat([rgb, ndvi], dim=1)
            
            if fusion_type== 'NDVI':
                input_mod = ndvi
            elif fusion_type == 'RGB':
                input_mod = rgb
            elif fusion_type == 'EarlyFusion':
                image_fusion = torch.cat([rgb,ndvi], dim=1)
                input_mod = image_fusion

            t0 = time.time()
            output,_ = model(input_mod)
            t1 = time.time()

            
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).float()

            metricts = compute_performance_metrics(predictions,mask)

            time_duration.append(t1-t0)
            recalls.append(metricts['recall'])
            precisions.append(metricts['precision'])
            f1_scores.append(metricts['f1'])
            ious.append(metricts['iou'])
            accuracies.append(metricts['acc'])


            input_mod = input_mod.detach().cpu().numpy()
            mask = mask.cpu().detach().numpy()
            probabilities = probabilities.cpu().detach().numpy()
            
            # SAVE DATA FOR FUTURE
            for i, (input,mask,pred,label) in enumerate(zip(input_mod,mask,probabilities,id)):
                with open(os.path.join(save_pred_mask_dir,label + '.pickle'), 'wb') as handle:
                    pickle.dump({'iou':metricts['iou'],'input':input,'mask':mask,'pred':pred}, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
    
    mean_duration = round(np.mean(time_duration),3)
   
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
    
    content =  f"| - SAVED MODEL- Model {parameters['model_name']} Dataset {parameters['dataset_name']} Duration(s): {mean_duration} Acc: {mean_accuracy_percent}%. Prec: {mean_precision_percent}%. Recall: {mean_recall_percent}%. F1: {mean_f1_score_percent}%. IoU: {mean_iou_percent}%"
    print(content)

    with open(overal_perfm, 'a') as appender:
        appender.write(content)
        appender.write('\n')




def select_set(dataset,batch_size=30):
    num_classes = 1

    assert dataset in ['t1','t2','t3','vg'],'Dataset not recognized'
    loader_val    = MVARGEMDataset(root=root_val, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes)
    loader_esac   = MVARGEMDataset(root=root_esac, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes)
    loader_qbaixo = MVARGEMDataset(root=root_qbaixo, set='altum', rgb_dir = 'images', mask_dir = 'masks', num_classes = num_classes)

    if dataset.lower() == 't1':
        train_dataset = torch.utils.data.ConcatDataset([loader_val, loader_esac])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(loader_qbaixo, batch_size=3, shuffle=False)
    elif dataset.lower() == 't2':
        train_dataset = torch.utils.data.ConcatDataset([loader_val, loader_qbaixo])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(loader_esac, batch_size=3, shuffle=False)
    elif dataset.lower() == 't3':
        train_dataset = torch.utils.data.ConcatDataset([loader_esac, loader_qbaixo])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(loader_val, batch_size=3, shuffle=False)
    elif dataset.lower() == 'vg':
        test_dataset = MAIZE_MVARGEMDataset( root=root_vg, set='test', rgb_dir = 'RGB', mask_dir = 'Masks', nir_dir = 'NIR', red_dir = 'RED', num_classes = num_classes)
        train_dataset = MAIZE_MVARGEMDataset( root=root_vg, set='train', rgb_dir = 'RGB', mask_dir = 'Masks', nir_dir = 'NIR', red_dir = 'RED', num_classes = num_classes)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False)
    return train_dataloader,test_dataloader




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


def main(log_file,save_pred_mask_dir,parameters):

    torch.manual_seed(0)
    np.random.seed(0)

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file, map_location={'cuda:0':'cuda:1'}))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')


    if os.path.exists(final_model_file):
        os.remove(final_model_file)

    train_dataloader,test_dataloader = select_set(parameters['dataset_name'])

    model = select_seg_model(parameters['model_name'],parameters['fusion_type'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    if args.gpu >= 0: model.cuda(args.gpu)

    best_f1 = 0.0
    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        
        acc_avg = train_epoch(epo, model, train_dataloader, optimizer, scheduler,parameters['fusion_type'])
        
        #validation(epo, model, val_loader)
        test_score = test_epoch(model, test_dataloader,parameters['fusion_type'])  #TEST DATASET
        
        # save check point model
        if test_score > best_f1:
            best_f1 = test_score
            torch.save(model.state_dict(), best_model_file)
        
        
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)

    # LOAD BEST MODEL AND SAVE PRED MASKS
    model.load_state_dict(torch.load(best_model_file,map_location=torch.device('cuda:0')))
    save_pred_mask(model, test_dataloader,save_pred_mask_dir,parameters)
        
    # Rename the checkpoint model file to the final model file
    os.rename(checkpoint_model_file, final_model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segnet with pytorch')
    parser.add_argument('--batch_size',  '-B',  type=int, default=batch_size)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=epochs)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()


    fusion_types  = ['RGB','NDVI']
    dataset_names = ['t1','t2','t3','vg']
    model_names   = ['segnet','deeplab'] #segnet or deeplabv3

    for fusion_type in fusion_types:
        for dataset_name in dataset_names:
            for model_name in model_names:
                save_name = os.path.join('weights',f'{model_name}_{dataset_name}_{fusion_type}')

                os.makedirs(save_name, exist_ok=True)
                checkpoint_model_file = os.path.join(save_name, 'tmp.pth')
                checkpoint_optim_file = os.path.join(save_name, 'tmp.optim')
                best_model_file       = os.path.join(save_name, 'best.pth')  
                final_model_file      = os.path.join(save_name, 'final.pth')
                log_file              = os.path.join(save_name, 'log.txt')
                save_pred_mask_dir    = os.path.join(save_name, 'pred_masks')

                if not os.path.isdir(save_pred_mask_dir):
                    os.makedirs(save_pred_mask_dir)
                print('| training %s on GPU #%d with pytorch' % (save_name, args.gpu))
                print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
                print('| model will be saved in: %s' % save_name)
                
                parameters = {'fusion_type':fusion_type,
                            'dataset_name':dataset_name,
                            'model_name':model_name,
                            'save_name':save_name
                            }


                main(log_file,save_pred_mask_dir,parameters)