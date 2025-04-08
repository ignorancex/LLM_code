#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--source_flag', type=str, help='choose RadImageNet, ImageNet or any of the MedMNIST datasets')
parser.add_argument('--target_flag', type=str, help='choose a MedMNIST')
parser.add_argument('--batch_size', type=int, help='batch size', default=128)
parser.add_argument('--epochs', type=int, help='number of epochs', default=45)
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
args = parser.parse_args()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision.transforms import v2, Lambda
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from PIL import Image
from medmnist import INFO
import radt
from tqdm import tqdm

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
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation auc increase.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def chest_transform(num_zeros):
    def chest_target_transform(y):
        # Convert label to a PyTorch tensor if it's not already
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float)
        
        # Append the specified number of zeros
        zeros = torch.zeros(num_zeros, dtype=torch.float)
        transformed_label = torch.cat([y, zeros])
        
        # Check if the original label is all zeros
        if torch.all(y == 0):
            transformed_label[len(y)] = 1.0  # Set the first appended zero to one
        
        return transformed_label
    
    return chest_target_transform

def load_data(target_flag):
    sources = ['chestmnist', 'pneumoniamnist', 'pathmnist', 'dermamnist', 'octmnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
    sources.remove(target_flag)
    # preprocessing
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomRotation(degrees=(0, 5)),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.RandomAutocontrast(),
        v2.RandomEqualize(),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    info = INFO[target_flag]
    if target_flag == 'pneumoniamnist' or target_flag == 'organamnist' or target_flag == 'organcmnist' or target_flag == 'organsmnist':
        n_classes = 69
    elif target_flag == 'chestmnist':
        n_classes = 56
    else:
        n_classes = 69 - len(info['label'])

    train_datasets = []
    val_datasets = []
    set_idx = 0
    for flag in sources:
        print(flag)
        info = INFO[flag]
        as_rgb = True if info['n_channels'] == 1 else False
        #one-hot
        current_set_idx = set_idx
        if flag == 'chestmnist':
            num_zeros = n_classes - 14
            target_transform = chest_transform(num_zeros)
            set_idx += 15
        elif flag == 'pneumoniamnist' and 'chestmnist' in sources:
            target_transform = Lambda(lambda y: torch.zeros(n_classes, dtype=torch.float).scatter_(dim=0, index=torch.tensor([14 if y == 0 else 6]), value=1))
        elif flag == 'organamnist' or flag == 'organsmnist' or flag == 'organcmnist':
            target_transform = Lambda(lambda y: torch.zeros(n_classes, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y+set_idx), value=1))
        else:
            target_transform = Lambda(lambda y, current_set_idx=current_set_idx: torch.zeros(n_classes, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y+ current_set_idx), value=1))
            set_idx += len(info['label'])

        target_data = np.load('/home/doju/.medmnist/' + flag +  '_224.npz')
        trainset = MyDataset(target_data["train_images"], target_data["train_labels"], transform=train_transform, target_transform=target_transform, as_rgb=as_rgb)
        valset = MyDataset(target_data["val_images"], target_data["val_labels"], transform=val_transform, target_transform=target_transform, as_rgb=as_rgb)
        train_datasets.append(trainset)
        val_datasets.append(valset)
    
    combined_traiset = ConcatDataset(train_datasets)
    combined_valset = ConcatDataset(val_datasets)
    return combined_traiset, combined_valset

def getAUC(y_true, y_score):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    auc = 0
    for i in range(y_score.shape[1]):
        label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
        auc += label_auc
    ret = auc / y_score.shape[1]

    return ret

def getACC(y_true, y_score, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    y_pre = y_score > threshold
    acc = 0
    for label in range(y_true.shape[1]):
        label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
        acc += label_acc
    ret = acc / y_true.shape[1]

    return ret

def my_test(model, data_loader, criterion, device):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = model(inputs.to(device))
            
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
            m = nn.Sigmoid()
            outputs = m(outputs).to(device)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            y_true = torch.cat((y_true, targets), 0)
            
        y_score = y_score.detach().cpu().numpy()
        print(y_score.sum(axis=0))
        y_true = y_true.detach().cpu().numpy()
        print(y_true.sum(axis=0))
        auc = getAUC(y_true, y_score)
        acc = getACC(y_true, y_score)

        return sum(total_loss)/len(total_loss), auc, acc
    
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    total_loss = []

    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = targets.to(torch.float32).to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    if scheduler:
        scheduler.step()

    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss

def pretrain(target_flag, lr, batch_size, epochs):
    seed=42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    with radt.run.RADTBenchmark() as run:
        info = INFO[target_flag]
        if target_flag == 'pneumoniamnist' or target_flag == 'organamnist' or target_flag == 'organcmnist' or target_flag == 'organsmnist':
            n_classes = 69
        elif target_flag == 'chestmnist':
            n_classes = 56
        else:
            n_classes = 69 - len(info['label'])
      
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = torchvision.models.resnet18(weights=None, num_classes=n_classes)
        net.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        trainset, valset = load_data(target_flag)

        print('collecting targets')
        # Get all targets
        targets = []
        for idx, (_, target) in tqdm(enumerate(trainset), total=len(trainset)):
            targets.append(np.argmax(target))
        targets = torch.tensor(targets)
        print('done collecting targets')
        
        # Compute samples weight (each sample should get its own weight)
        class_sample_count = torch.tensor([(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets])

        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = DataLoader(trainset,
            batch_size=batch_size,
            sampler=sampler
        )
        
        val_loader = DataLoader(valset,
            batch_size=batch_size,
            shuffle=False
        )

        for key, value in vars(args).items():
            run.log_param(key, value)

        print('==> Building and training model...')
        path=f'/mnt/models/doju_pre-trained_for/{target_flag}.pt'
        early_stopping = EarlyStopping(patience=20, verbose=True, path=path)
        for epoch in range(epochs):  # loop over the dataset multiple times
            print(epoch)
            train_loss = train_epoch(net, train_loader, criterion, optimizer, scheduler, device)
            loss, train_auc, train_acc = my_test(net, train_loader, criterion, device)
            print('train loss, auc:', train_loss, train_auc)
            val_loss, val_auc, val_acc = my_test(net, val_loader, criterion, device)
            print('val loss, auc:', val_loss, val_auc)
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        run.log_metric("val_auc", val_auc)
        run.log_metric("stopped_at", epoch)
        print(val_auc)
        print("Finished Training")

pretrain(args.target_flag, args.lr, args.batch_size, args.epochs)