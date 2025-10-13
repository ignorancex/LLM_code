import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_calibration_error

def margin_topk_accuracy(predictions, labels, k=10):
    # get the predcitions with the highest probabilities
    values, indices = torch.topk(torch.nn.functional.softmax(predictions, dim=1),k=2, dim=1)

    # get difference of first and second
    values = values[:,0] - values[:,1] 
    values, idx= torch.topk(values,k)

    acc = torch.sum(indices[idx].cpu()[:,0] == labels.cpu()[idx.cpu()]) / k

    return acc


def topk_accuracy(predictions, labels, k=10):


    # get the predcitions with the highest probabilities
    values, indices = torch.max(torch.nn.functional.softmax(predictions, dim=1), dim=1) 
    values, idx= torch.topk(values,k)

    acc = torch.sum(indices[idx].cpu() == labels.cpu()[idx.cpu()]) / k

    return acc

def calculate_metrics(outputs, labels, metrics_to_log: dict, pretext= "",n_labels=7,k=20):

    class_predictions = torch.nn.functional.softmax(outputs,dim=1)

    if len(labels) < k:
        k = len(labels)
    predictions = torch.argmax(class_predictions, dim=1)
    if f"{pretext}top20_accuracy" in metrics_to_log:
        metrics_to_log[f"{pretext}top20_accuracy"] += topk_accuracy(class_predictions, labels, k=k) 
    if f"{pretext}accuracy" in metrics_to_log:
        metrics_to_log[f"{pretext}accuracy"] += torch.sum(predictions == labels).item() / len(labels)
    if f"{pretext}margin_top20" in metrics_to_log:
        metrics_to_log[f"{pretext}margin_top20"] += margin_topk_accuracy(class_predictions, labels, k=k)
    if f"{pretext}brier_score" in metrics_to_log:
        metrics_to_log[f"{pretext}brier_score"] += brier_score(outputs, labels).item()
    if f"{pretext}ece" in metrics_to_log:
        metrics_to_log[f"{pretext}ece"] += multiclass_calibration_error(outputs, labels, n_labels).item()
    
    return metrics_to_log

def brier_score(predictions, labels):

    predictions = torch.nn.functional.softmax(predictions, dim=1)
    one_hot_labels = torch.zeros_like(predictions)
    one_hot_labels[torch.arange(labels.shape[0]), labels] = 1.0

    return torch.mean(torch.sum((predictions - one_hot_labels)**2, dim=1))

