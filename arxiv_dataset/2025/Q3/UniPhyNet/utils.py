import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split
from models import *
from dataset import *


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    all_targets, all_preds = [], []
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        # inputs, labels_a, labels_b, lam = mixup_data(inputs, labels,alpha = 0.1)
        optimizer.zero_grad()
        outputs = model(inputs)
        # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    accuracy, precision, recall, f1 = calculate_metrics(all_targets, all_preds)
    return total_loss / len(train_loader), accuracy, precision, recall, f1

def evaluate(model, val_loader, criterion, device):
    model.eval()
    all_targets, all_preds = [], []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    accuracy, precision, recall, f1 = calculate_metrics(all_targets, all_preds)
    return total_loss / len(val_loader), accuracy, precision, recall, f1

def visualize_results(history, title):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 10-Fold Cross-Validation Function
def cross_validate_10fold(num_classes,type, data ,kernels, num_feature_maps, res_blocks, in_channels,learning_rate = 1e-3):
    if num_classes == 2:
        task = 'binary'
    elif num_classes == 3:
        task = 'ternary'
        
    if type == 'eeg':
        dataset_with_aug = EEGDataset(data, task=task, augment=True)
        dataset_without_aug = EEGDataset(data, task=task, augment=False)
        samples = 2560
    elif type == 'ecg':
        dataset_with_aug = ECGDataset(data, task=task, augment=True)
        dataset_without_aug = ECGDataset(data, task=task, augment=False)
        samples = 5120
    elif type == 'eda':
        dataset_with_aug = EDADataset(data, task=task, augment=True)
        dataset_without_aug = EDADataset(data, task=task, augment=False)
        samples = 1280
        
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    kernels_str = '_'.join(map(str, kernels))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/10fold_{type}_training_log_{task}_k{kernels_str}_f{num_feature_maps}_r{res_blocks}_in{in_channels}_{current_time}.csv"

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold','Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Val Loss', 'Val Acc', 'Val F1','Val Precision','Val Recall'])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset_without_aug)), dataset_without_aug.y)):
            train_subset = Subset(dataset_with_aug, train_idx)
            val_subset = Subset(dataset_without_aug, val_idx)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

            model = UniPhyNetSingle(kernels=kernels,samples = samples, num_feature_maps=num_feature_maps, res_blocks=res_blocks, in_channels=in_channels, num_classes=num_classes).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
            # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1)


            for epoch in range(60):
                train_loss, train_acc, train_prec, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
                print(f'Fold {fold + 1}, Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                writer.writerow([fold+1,epoch+1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, val_prec, val_recall])

                scheduler.step(val_loss)
            

        

    visualize_results(history, '10-Fold Cross-Validation')


# Leave-One-Subject-Out Cross-Validation Function
def cross_validate_loso(num_classes,type, data ,kernels, num_feature_maps, res_blocks, in_channels,learning_rate = 1e-3):
    if num_classes == 2:
        task = 'binary'
    elif num_classes == 3:
        task = 'ternary'
        
    if type == 'eeg':
        dataset_with_aug = EEGDataset(data, task=task, augment=True)
        dataset_without_aug = EEGDataset(data, task=task, augment=False)
        samples = 2560
    elif type == 'ecg':
        dataset_with_aug = ECGDataset(data, task=task, augment=True)
        dataset_without_aug = ECGDataset(data, task=task, augment=False)
        samples = 5120
    elif type == 'eda':
        dataset_with_aug = EDADataset(data, task=task, augment=True)
        dataset_without_aug = EDADataset(data, task=task, augment=False)
        samples = 1280
        
    logo = LeaveOneGroupOut()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    kernels_str = '_'.join(map(str, kernels))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/loso_{type}_training_log_{task}_k{kernels_str}_f{num_feature_maps}_r{res_blocks}_in{in_channels}_{current_time}.csv"

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold','Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Val Loss', 'Val Acc', 'Val F1','Val Precision','Val Recall'])
        for fold, (train_idx, val_idx) in enumerate(logo.split(np.zeros(len(dataset_without_aug)), dataset_without_aug.y, dataset_without_aug.groups)):
            train_subset = Subset(dataset_with_aug, train_idx)
            val_subset = Subset(dataset_without_aug, val_idx)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

            model = UniPhyNetSingle(kernels=kernels,samples = samples, num_feature_maps=num_feature_maps, res_blocks=res_blocks, in_channels=in_channels, num_classes=num_classes).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
            # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1)

            for epoch in range(60):
                train_loss, train_acc, train_prec, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
                print(f'Fold {fold + 1}, Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                writer.writerow([fold+1,epoch+1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, val_prec, val_recall])

                scheduler.step(val_loss)

    visualize_results(history, '10-Fold Cross-Validation')



def train_test_split_pipeline(num_classes,type, data ,kernels, num_feature_maps, res_blocks, in_channels,learning_rate = 1e-3,lr_scheduler = False):
    if num_classes == 2:
        task = 'binary'
    elif num_classes == 3:
        task = 'ternary'
        
    if type == 'eeg':
        dataset = EEGDataset(data, task=task, augment=True)
        val_dataset = EEGDataset(data, task=task, augment=False)
        samples = 2560
    elif type == 'ecg':
        dataset = ECGDataset(data, task=task, augment=True)
        val_dataset = ECGDataset(data, task=task, augment=False)
        samples = 5120
    elif type == 'eda':
        dataset = EDADataset(data, task=task, augment=True)
        val_dataset = EDADataset(data, task=task, augment=False)
        samples = 1280
        
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.10, random_state=42)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UniPhyNetSingle(kernels=kernels,samples = samples, num_feature_maps=num_feature_maps, res_blocks=res_blocks, in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Create a filename based on parameters and current datetime
    kernels_str = '_'.join(map(str, kernels))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/{type}_training_log_{task}_k{kernels_str}_f{num_feature_maps}_r{res_blocks}_in{in_channels}_{current_time}.csv"

    # Open a CSV file to write the log
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Val Loss', 'Val Acc', 'Val F1','Val Precision','Val Recall'])

        for epoch in range(100):  # Adjust the number of epochs if necessary
            train_loss, train_acc, train_prec, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Log the results to the CSV file
            writer.writerow([epoch+1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, val_prec, val_recall])

            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            if lr_scheduler:
                scheduler.step(val_loss)

    visualize_results(history, 'Training with Augmentation, Early Stopping, and Adaptive LR')

    return model, history

def train_multi(model, train_loader, criterion, optimizer, device):
    model.train()
    all_targets, all_preds = [], []
    total_loss = 0.0
    for (eeg_data,ecg_data,eda_data), labels in train_loader:
        eeg_data,ecg_data,eda_data, labels = eeg_data.to(device),ecg_data.to(device),eda_data.to(device), labels.to(device).squeeze()
        # inputs, labels_a, labels_b, lam = mixup_data(inputs, labels,alpha = 0.1)
        optimizer.zero_grad()
        outputs = model(eeg_data,ecg_data,eda_data)
        # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    accuracy, precision, recall, f1 = calculate_metrics(all_targets, all_preds)
    return total_loss / len(train_loader), accuracy, precision, recall, f1

def evaluate_multi(model, val_loader, criterion, device):
    model.eval()
    all_targets, all_preds = [], []
    total_loss = 0.0
    with torch.no_grad():
        for (eeg_data,ecg_data,eda_data), labels in val_loader:
            eeg_data,ecg_data,eda_data, labels = eeg_data.to(device),ecg_data.to(device),eda_data.to(device), labels.to(device).squeeze()
            outputs = model(eeg_data,ecg_data,eda_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    accuracy, precision, recall, f1 = calculate_metrics(all_targets, all_preds)
    return total_loss / len(val_loader), accuracy, precision, recall, f1

def train_double(model, train_loader, criterion, optimizer, device):
    model.train()
    all_targets, all_preds = [], []
    total_loss = 0.0
    for (eeg_data,ecg_data), labels in train_loader:
        eeg_data,ecg_data, labels = eeg_data.to(device),ecg_data.to(device), labels.to(device).squeeze()
        # inputs, labels_a, labels_b, lam = mixup_data(inputs, labels,alpha = 0.1)
        optimizer.zero_grad()
        outputs = model(eeg_data,ecg_data)
        # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    accuracy, precision, recall, f1 = calculate_metrics(all_targets, all_preds)
    return total_loss / len(train_loader), accuracy, precision, recall, f1

def evaluate_double(model, val_loader, criterion, device):
    model.eval()
    all_targets, all_preds = [], []
    total_loss = 0.0
    with torch.no_grad():
        for (eeg_data,ecg_data), labels in val_loader:
            eeg_data,ecg_data, labels = eeg_data.to(device),ecg_data.to(device), labels.to(device).squeeze()
            outputs = model(eeg_data,ecg_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    accuracy, precision, recall, f1 = calculate_metrics(all_targets, all_preds)
    return total_loss / len(val_loader), accuracy, precision, recall, f1

def multi_cross_validate_10fold(num_classes,type, data ,kernels,samples, num_feature_maps, res_blocks, in_channels,learning_rate = 1e-3):
    if num_classes == 2:
        task = 'binary'
    elif num_classes == 3:
        task = 'ternary'
    
 
    if len(type) == 2:
        modelclass = UniPhyNetDouble
        train = train_double
        evaluate = evaluate_double
    elif len(type) == 3:
        modelclass = UniPhyNetMulti
        train = train_multi
        evaluate = evaluate_multi
    
    dataset_with_aug = MultiDatasetFeatureFusion(data, task=task, augment=True,type = type)
    dataset_without_aug = MultiDatasetFeatureFusion(data, task=task, augment=False,type =type) 

        
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    kernels_str = '_'.join(map(str, kernels))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/10fold_{type}_training_log_{task}_k{kernels_str}_f{num_feature_maps}_r{res_blocks}_in{in_channels}_{current_time}.csv"

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold','Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Val Loss', 'Val Acc', 'Val F1','Val Precision','Val Recall'])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset_without_aug)), dataset_without_aug.y)):
            train_subset = Subset(dataset_with_aug, train_idx)
            val_subset = Subset(dataset_without_aug, val_idx)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

            model = modelclass(kernels_list=kernels,samples_list = samples, num_feature_maps_list=num_feature_maps, res_blocks_list=res_blocks, in_channels_list=in_channels, num_classes=num_classes).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
            # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1)


            for epoch in range(60):
                train_loss, train_acc, train_prec, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
                print(f'Fold {fold + 1}, Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                writer.writerow([fold+1,epoch+1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, val_prec, val_recall])

                scheduler.step(val_loss)
            

        

    visualize_results(history, '10-Fold Cross-Validation')