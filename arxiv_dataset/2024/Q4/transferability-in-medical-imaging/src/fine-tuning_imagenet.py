#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--source_flag', type=str, help='choose densenet, efficientnet, googlenet, inception, mobilenet, vgg, convnext, shufflenet')
parser.add_argument('--target_flag', type=str, help='choose a MedMNIST')
parser.add_argument('--batch_size', type=int, help='batch size', default=32)
parser.add_argument('--epochs', type=int, help='number of epochs', default=30)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--wd', type=float, help='weight decay', default=0.001)
args = parser.parse_args()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from PIL import Image
from medmnist import INFO
import radt

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None, as_rgb=False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index].astype(int)
        x = Image.fromarray(x)

        if self.as_rgb:
            x = x.convert("RGB")
        
        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_auc, model):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation auc increase.'''
        if self.verbose:
            self.trace_func(f'Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc

def load_data(source_flag, target_flag):
    info = INFO[target_flag]
    as_rgb = True if info['n_channels'] == 1 else False
    # preprocessing
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomRotation(degrees=(0, 5)),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.RandomAutocontrast(),
        v2.RandomEqualize(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
    ])
    
    target_data = np.load('/mnt/share/data/medmnist_similarity/' + target_flag +  '_fine-tune_224.npz')
    trainset = MyDataset(target_data["train_imgs_fold1"], target_data["train_labels_fold1"], transform=train_transform, as_rgb=as_rgb)
    valset = MyDataset(target_data["val_imgs_fold1"], target_data["val_labels_fold1"], transform=test_transform, as_rgb=as_rgb)

    return trainset, valset

def load_test_data(source_flag, target_flag):
    info = INFO[target_flag]
    as_rgb = True if info['n_channels'] == 1 else False
    # preprocessing
    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_data = np.load('/home/doju/.medmnist/' + target_flag +  '_224.npz')
    testset = MyDataset(target_data["test_images"], target_data["test_labels"], transform=test_transform, as_rgb=as_rgb)

    return testset

def getAUC(y_true, y_score, task):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret

def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret

def my_test(model, data_loader, task, criterion, device):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            y_true = torch.cat((y_true, targets), 0)
            
        y_score = y_score.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)

        return sum(total_loss)/len(total_loss), auc, acc
    
def train_epoch(model, train_loader, task, criterion, optimizer, scheduler, device):
    total_loss = []

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    scheduler.step()

    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss

def fine_tune(source_flag, target_flag, lr, momentum, wd, batch_size, epochs):
    seed=42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    with radt.run.RADTBenchmark() as run:
        target_info = INFO[target_flag]
        n_classes = len(target_info['label'])
        task = target_info['task']
        
        # Log parameters to mlflow
        for key, value in vars(args).items():
            run.log_param(key, value)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if source_flag == 'densenet':
            net = torchvision.models.densenet121(weights='IMAGENET1K_V1')
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'efficientnet':
            net = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            num_ftrs = net.classifier[-1].in_features
            net.classifier = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'googlenet':
            net = torchvision.models.googlenet(weights='IMAGENET1K_V1')
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'mnasnet':
            net = torchvision.models.mnasnet1_0(weights='IMAGENET1K_V1')
            num_ftrs = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'mobilenet':
            net = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            num_ftrs = net.classifier[-1].in_features
            net.classifier[-1] =nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'vgg':
            net = torchvision.models.vgg11(weights='IMAGENET1K_V1')
            num_ftrs = net.classifier[-1].in_features
            net.classifier[-1] =nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'convnext':
            net = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
            num_ftrs = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'shufflenet':
            net = torchvision.models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, n_classes)

        net.to(device)

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=False if momentum == 0.0 else True, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        trainset, valset = load_data(source_flag, target_flag)

        train_loader = DataLoader(trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        for key, value in vars(args).items():
            run.log_param(key, value)
        print('==> Building and training model...')
        path=f'/mnt/share/models/doju_sim_finetuned_imnet/{source_flag}_{target_flag}_{lr}_{momentum}_{wd}_{batch_size}.pt'
        early_stopping = EarlyStopping(patience=50, verbose=True, path=path)
        for epoch in range(epochs):  # loop over the dataset multiple times
            train_loss = train_epoch(net, train_loader, task, criterion, optimizer, scheduler, device)
            loss, train_auc, train_acc = my_test(net, train_loader, task, criterion, device)
            print('train loss, auc:', train_loss, train_auc)
            val_loss, val_auc, val_acc = my_test(net, val_loader, task, criterion, device)
            print('val loss, auc:', val_loss, val_auc)
            early_stopping(val_auc, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        if source_flag == 'densenet':
            ft_model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.classifier.in_features
            ft_model.classifier = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'efficientnet':
            ft_model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.classifier[-1].in_features
            ft_model.classifier = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'googlenet':
            ft_model = torchvision.models.googlenet(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.fc.in_features
            ft_model.fc = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'mnasnet':
            ft_model = torchvision.models.mnasnet1_0(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.classifier[1].in_features
            ft_model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'mobilenet':
            ft_model = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.classifier[-1].in_features
            ft_model.classifier[-1] =nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'vgg':
            ft_model = torchvision.models.vgg11(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.classifier[-1].in_features
            ft_model.classifier[-1] =nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'convnext':
            ft_model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.classifier[-1].in_features
            ft_model.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        elif source_flag == 'shufflenet':
            ft_model = torchvision.models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
            num_ftrs = ft_model.fc.in_features
            ft_model.fc = nn.Linear(num_ftrs, n_classes)
        ft_model.load_state_dict(torch.load(path))
        ft_model.to(device)
        val_loss, val_auc, val_acc = my_test(ft_model, val_loader, task, criterion, device)
        testset = load_test_data(source_flag, target_flag)
        test_loader = DataLoader(testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        test_loss, test_auc, test_acc = my_test(ft_model, test_loader, task, criterion, device)
        run.log_metric("val_auc", val_auc)
        run.log_metric("val_loss", val_loss)
        run.log_metric("val_acc", val_acc)
        run.log_metric("stopped_at", epoch)
        run.log_metric("test_auc", test_auc)
        run.log_metric("test_loss", test_loss)
        run.log_metric("test_acc", test_acc)
        print(val_auc)
        print("Finished Training")

fine_tune(args.source_flag, args.target_flag, args.lr, args.momentum, args.wd, args.batch_size, args.epochs)