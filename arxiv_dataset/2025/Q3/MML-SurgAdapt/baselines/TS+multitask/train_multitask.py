import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from sklearn.metrics import average_precision_score
from task_specific_baselines.cholec80 import Cholec80
from task_specific_baselines.endoscapes import Endoscapes
from task_specific_baselines.cholect50 import CholecT50
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing, InterpolationMode

import os
import time
from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from statistics import mean
import ivtmetrics
from datetime import datetime
import json
import random

import torch
from torch.cuda.amp import GradScaler, autocast  # type: ignore
import torch.nn.functional
from torch.optim import lr_scheduler

from utils import AverageMeter, add_weight_decay, mAP
import warnings

def process_cholec80(true,pred,video_ids):

    true = np.round(true)

    unique_ids = np.unique(video_ids)
    video_f1s = {}

    for vid in unique_ids:
        indices = np.where(video_ids==vid)[0]
        y_true_video = true[indices]  
        y_pred_video = pred[indices]

        y_pred_video = np.argmax(y_pred_video,axis=1)
        y_true_video = np.argmax(y_true_video,axis=1)

        f1 = f1_score(y_true_video, y_pred_video, average="macro",labels=np.unique(y_true_video))*100

        video_f1s[vid] = f1

    f1_score_reg = mean(video_f1s.values())

    return f1_score_reg, video_f1s

def resolve_nan(classwise):
        classwise[classwise==-0.0] = np.nan
        return classwise

def process_cholect50(true,pred,video_ids):

    true = np.round(true)

    unique_vids = np.unique(video_ids)

    ap_i_list = []
    ap_v_list = []
    ap_t_list = []
    ap_iv_list = []
    ap_it_list = []
    ap_ivt_list = []

    for vid in unique_vids:
            indices = np.where(video_ids == vid)[0]
            ivt_labels = true[indices]
            ivt_preds = pred[indices]

            filter = ivtmetrics.Disentangle()

            i_labels = filter.extract(inputs=ivt_labels, component="i")
            v_labels = filter.extract(inputs=ivt_labels, component="v")
            t_labels = filter.extract(inputs=ivt_labels, component="t")
            iv_labels = filter.extract(inputs=ivt_labels, component="iv")
            it_labels = filter.extract(inputs=ivt_labels, component="it")

            i_preds = filter.extract(inputs=ivt_preds, component="i")
            v_preds = filter.extract(inputs=ivt_preds, component="v")
            t_preds = filter.extract(inputs=ivt_preds, component="t")
            iv_preds = filter.extract(inputs=ivt_preds, component="iv")
            it_preds = filter.extract(inputs=ivt_preds, component="it")

            ap_i = average_precision_score(i_labels,i_preds,average=None)*100
            ap_v = average_precision_score(v_labels,v_preds,average=None)*100
            ap_t = average_precision_score(t_labels,t_preds,average=None)*100
            ap_iv = average_precision_score(iv_labels,iv_preds,average=None)*100
            ap_it = average_precision_score(it_labels,it_preds,average=None)*100
            ap_ivt = average_precision_score(ivt_labels,ivt_preds,average=None)*100

            ap_i = resolve_nan(ap_i)
            ap_v = resolve_nan(ap_v)
            ap_t = resolve_nan(ap_t)
            ap_iv = resolve_nan(ap_iv)
            ap_it = resolve_nan(ap_it)
            ap_ivt = resolve_nan(ap_ivt)

            ap_i_list.append(ap_i.reshape([1,-1]))
            ap_v_list.append(ap_v.reshape([1,-1]))
            ap_t_list.append(ap_t.reshape([1,-1]))
            ap_iv_list.append(ap_iv.reshape([1,-1]))
            ap_it_list.append(ap_it.reshape([1,-1]))
            ap_ivt_list.append(ap_ivt.reshape([1,-1]))
    
    ap_i_list = np.concatenate(ap_i_list,axis=0)
    ap_i_list = np.nanmean(ap_i_list,axis=0)
    map_i = np.nanmean(ap_i_list)
    ap_v_list = np.concatenate(ap_v_list,axis=0)
    ap_v_list = np.nanmean(ap_v_list,axis=0)
    map_v = np.nanmean(ap_v_list)
    ap_t_list = np.concatenate(ap_t_list,axis=0)
    ap_t_list = np.nanmean(ap_t_list,axis=0)
    map_t = np.nanmean(ap_t_list)
    ap_iv_list = np.concatenate(ap_iv_list,axis=0)
    ap_iv_list = np.nanmean(ap_iv_list,axis=0)
    map_iv = np.nanmean(ap_iv_list)
    ap_it_list = np.concatenate(ap_it_list,axis=0)
    ap_it_list = np.nanmean(ap_it_list,axis=0)
    map_it = np.nanmean(ap_it_list)
    ap_ivt_list = np.concatenate(ap_ivt_list,axis=0)
    ap_ivt_list = np.nanmean(ap_ivt_list,axis=0)
    map_ivt = np.nanmean(ap_ivt_list)

    aps = {
        "AP_i" : map_i,
        "AP_v" : map_v,
        "AP_t" : map_t,
        "AP_iv" : map_iv,
        "AP_it" : map_it,
        "AP_ivt" : map_ivt
    }

    return aps

def process_endo(true,pred,video_ids):

    true = np.round(true)

    num_classes = true.shape[1]

    per_class_map = {}

    for label in range(num_classes):
        avg_precision = average_precision_score(
            true[:, label], pred[:, label]
        )
        per_class_map[f"C{label}"] = avg_precision * 100.0
    
    mean_ap = mean(per_class_map.values())


    return mean_ap, per_class_map

def save_results(a,b,c,test,dir):

    cholec80_data = {
        "F1_score" : a[0],
        "Per_video_f1" : a[1],
    }

    endo_data = {
        "mAP" : b[0],
        "mAP_per_class" : b[1],
    }

    cholect50_data = {
        "AP" : c,
    }

    data = {
        "Test" : test,
        "Cholec80" : cholec80_data,
        "Endoscapes" : endo_data,
        "CholecT50" : cholect50_data
    }

    folder_name = f"results/{dir}"
    os.makedirs(f"results/{dir}",exist_ok=True)

    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    if test == True:
        file_name = f"{folder_name}/result_{timestamp}_test.json"
    else:
        file_name = f"{folder_name}/result_{timestamp}.json"

    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("Results saved successfully")
    except Exception as e:
        print(f"Error in saving results : {e}")

# Custom Multi-Head Model
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes1, num_classes2, num_classes3):
        super(MultiTaskModel, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the classification head
        
        self.head1 = nn.Linear(2048, num_classes1)  # Multi-class classification
        self.head2 = nn.Linear(2048, num_classes2)  # Multi-label classification
        self.head3 = nn.Linear(2048, num_classes3)  # Multi-label classification
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head1(features), self.head2(features), self.head3(features)

def train_epoch(model, dataloaders, optimizers, criterions, device):
    model.train()
    total_losses = [0.0, 0.0, 0.0]

    # Define how often to print progress
    print_interval = 50

    for i, (data1, data2, data3) in enumerate(zip(*dataloaders)):
        x1, y1 = data1[0].to(device), data1[1].to(device)
        x2, y2 = data2[0].to(device), data2[1].to(device)
        x3, y3 = data3[0].to(device), data3[1].to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        out1, out2, out3 = model(torch.cat([x1, x2, x3], dim=0))
        loss1 = criterions[0](out1[:len(y1)], y1)
        loss2 = criterions[1](out2[len(y1):len(y1) + len(y2)], y2)
        loss3 = criterions[2](out3[-len(y3):], y3)

        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizers[0].step()
        optimizers[1].step()
        optimizers[2].step()

        total_losses[0] += loss1.item()
        total_losses[1] += loss2.item()
        total_losses[2] += loss3.item()

        # Print progress after every `print_interval` steps
        if (i + 1) % print_interval == 0:
            print(f"Iteration {i + 1}/{len(dataloaders[0])} - "
                  f"Loss1: {total_losses[0] / (i + 1):.4f}, "
                  f"Loss2: {total_losses[1] / (i + 1):.4f}, "
                  f"Loss3: {total_losses[2] / (i + 1):.4f}")

    return [loss / len(dataloaders[0]) for loss in total_losses]

def validate_epoch_on_loss(model, dataloaders, criterions, device, test):
    model.eval()
    total_losses = [0.0, 0.0, 0.0]

    # Define how often to print progress
    print_interval = 50

    out_1, out_2, out_3 = [], [], []
    y_1, y_2, y_3 = [], [], []
    v_1, v_2, v_3 = [], [], []
    
    with torch.no_grad():
        for i, d in enumerate(dataloaders):
            for j, (image, label, vid) in enumerate(d):
                image = image.cuda()
                label = label.cuda()

                out = torch.sigmoid(model(image)[i])
                loss = criterions[i](out,label)

                total_losses[i] += loss.item()

                # Print progress after every `print_interval` steps
                if (j + 1) % print_interval == 0:
                    print(f"Validation Iteration {j + 1}/{len(d)} - "
                        f"Loss1: {total_losses[0] / (j + 1):.4f}, "
                        f"Loss2: {total_losses[1] / (j + 1):.4f}, "
                        f"Loss3: {total_losses[2] / (j + 1):.4f}")

                if i == 0:
                    out_1.append(out.cpu().detach().numpy())
                    y_1.append(label.cpu().detach().numpy())
                    v_1.append(vid)
                if i == 1:
                    out_2.append(out.cpu().detach().numpy())
                    y_2.append(label.cpu().detach().numpy())
                    v_2.append(vid)
                if i == 2:
                    out_3.append(out.cpu().detach().numpy())
                    y_3.append(label.cpu().detach().numpy())
                    v_3.append(vid)

    y_1 = np.concatenate(y_1, axis=0)
    out_1 = np.concatenate(out_1, axis=0)
    v_1 = np.concatenate([np.array(sublist) for sublist in v_1])
    y_2 = np.concatenate(y_2, axis=0)
    out_2 = np.concatenate(out_2, axis=0)
    v_2 = np.concatenate([np.array(sublist) for sublist in v_2])
    y_3 = np.concatenate(y_3, axis=0)
    out_3 = np.concatenate(out_3, axis=0)
    v_3 = np.concatenate([np.array(sublist) for sublist in v_3])

    # print(f"out1 : {out_1.shape}")
    # print(f"out2 : {out_2.shape}")
    # print(f"out3 : {out_3.shape}")
    # print(f"y1 : {y_1.shape}")
    # print(f"y2 : {y_2.shape}")
    # print(f"y3 : {y_3.shape}")
    # print(f"v1 : {v_1.shape}")
    # print(f"v2 : {v_2.shape}")
    # print(f"v3 : {v_3.shape}")

    a = process_cholec80(y_1, out_1, v_1)
    b = process_endo(y_2, out_2, v_2)
    c = process_cholect50(y_3, out_3, v_3)
    save_results(a, b, c, test, 'multitask')

    return [loss / len(dataloaders[i]) for i,loss in enumerate(total_losses)]

def main():
    device = 'cuda'
    torch.autograd.set_detect_anomaly(True)
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    # Model initialization
    model = MultiTaskModel(num_classes1=7, num_classes2=3, num_classes3=100).to(device)

    train_preprocess = transforms.Compose([
            transforms.Resize((224,224),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    val_preprocess = transforms.Compose([
            transforms.Resize((224,224),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    train_data1 = Cholec80('train',train_preprocess)
    train_data2 = Endoscapes('train',train_preprocess)
    train_data3 = CholecT50('train',train_preprocess)
    val_data1 = Cholec80('val',val_preprocess)
    val_data2 = Endoscapes('val',val_preprocess)
    val_data3 = CholecT50('val',val_preprocess)
    test_data1 = Cholec80('test',val_preprocess)
    test_data2 = Endoscapes('test',val_preprocess)
    test_data3 = CholecT50('test',val_preprocess)

    batch_size = 32  # Adjust this based on your GPU memory
    num_workers = 4  # Number of workers for data loading

    # Train DataLoaders
    train_loader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader2 = DataLoader(train_data2, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader3 = DataLoader(train_data3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Validation DataLoaders
    val_loader1 = DataLoader(val_data1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader2 = DataLoader(val_data2, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader3 = DataLoader(val_data3, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Test DataLoaders
    test_loader1 = DataLoader(test_data1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader3 = DataLoader(test_data3, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Dataloaders (replace with your actual DataLoader objects)
    dataloaders = [train_loader1, train_loader2, train_loader3]
    val_dataloaders = [val_loader1, val_loader2, val_loader3]
    test_dataloaders = [test_loader1, test_loader2, test_loader3]

    # Loss functions and optimizers
    criterions = [
        nn.CrossEntropyLoss(),
        nn.BCEWithLogitsLoss(),
        nn.BCEWithLogitsLoss()
    ]
    optimizers = [
        optim.Adam(list(model.backbone.parameters()) + list(model.head1.parameters()), lr=1e-4),
        optim.Adam(list(model.backbone.parameters()) + list(model.head2.parameters()), lr=1e-4),
        optim.Adam(list(model.backbone.parameters()) + list(model.head3.parameters()), lr=1e-4)
    ]

    best_loss = float('inf')
    best_model_state = None
    epochs = 50

    for epoch in range(epochs):
        train_losses = train_epoch(model, dataloaders, optimizers, criterions, device)
        val_losses = validate_epoch_on_loss(model, val_dataloaders, criterions, device, False)

        print(f"Epoch {epoch+1}/{epochs}: Train Losses {train_losses}, Val Losses {val_losses}")

        # Validation loss is the sum of losses for all tasks
        total_val_loss = sum(val_losses)
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            best_model_state = model.state_dict()

    print("Training complete. Best validation loss:", best_loss)

    # Save the best model
    torch.save(best_model_state, "best_model_multitask.pth")

    # Load the best model for testing
    model.load_state_dict(torch.load("best_model_multitask.pth"))
    test_losses = validate_epoch_on_loss(model, test_dataloaders, criterions, device, True)
    print("Test Losses:", test_losses)
    print("Total Test Loss:", sum(test_losses))

if __name__ == "__main__":
    main()
