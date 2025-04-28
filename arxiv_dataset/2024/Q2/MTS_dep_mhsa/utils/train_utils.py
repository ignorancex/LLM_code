import torch
import copy
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import time
import torch.nn.functional as F
from ct_data_loader import CtImages
import argparse
import pickle
from IPython.display import clear_output, display, HTML
from matplotlib import pyplot as plt
import torchvision
from tqdm import tqdm
from collections import defaultdict
from utils.early_stop import EarlyStopping
from utils.metric_evaluate import *
from torch.cuda.amp import autocast as autocast, GradScaler
from einops import rearrange, repeat, reduce

def validation_run(data_loader, device, model, criterion, val_or_test="Val"):
    running_loss = 0.0
    y_true_list = []
    y_pred_list = []
    y_abnormal = 0
    y_normal = 0
    stream = tqdm(data_loader)
    metric_monitor_valid = MetricMonitor()

    for batch_idx, (videos, y_true, patient_id, _) in enumerate(stream):
        videos = videos.to(device=device).float()
        y_true_list.extend(y_true.flatten().cpu().tolist())
        y_true = y_true.to(device=device).long()
        y_abnormal += y_true.sum().cpu()
        y_normal += torch.Tensor([videos.size()[0] - y_true.sum()]).cpu()

        with autocast():
            reg_out = model(videos)
            y_pred_list.extend(torch.argmax(F.softmax(reg_out.cpu().float(), dim=1), dim=1).tolist())
            loss = criterion(reg_out, y_true)
        
        running_loss += loss.item()
        record_dict = calculate_metrics(torch.Tensor(torch.argmax(F.softmax(reg_out.cpu().float(), dim=1), dim=1).tolist()),
                                        y_true.cpu())
        cnt_metric = len(y_true)

        metric_monitor_valid.update('Loss', loss.item(), cnt_metric)
        metric_monitor_valid.update('Accuracy', record_dict['acc'], cnt_metric)

        stream.set_description(
            "{val_or_test}. {metric_monitor}".format(
                val_or_test = val_or_test,
                metric_monitor=metric_monitor_valid)
        )
    print("\t\tlabel distribution: [ normal: abnormal ]: [", y_normal,":",y_abnormal, "]")
    return metric_monitor_valid.metrics['Loss']['avg'], metric_monitor_valid.metrics['Accuracy']['avg']