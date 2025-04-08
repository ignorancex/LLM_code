#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--target_flag', type=str, help='choose a MedMNIST')
args = parser.parse_args()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from PIL import Image
from medmnist import INFO
import radt
from tqdm import tqdm
import os

class MyDataset(Dataset):
    def __init__(self, dataset_dir, folder, csv_file, keep_columns, transform=None):
        self.dataset_dir = os.path.join(dataset_dir, folder)
        self.transform = transform or ToTensor()

        # read csv file
        if isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            self.df = pd.read_csv(csv_file)
            self.df = self.df[keep_columns]
        """
        # imgnet pre-processing
        for col in keep_columns:
            if col == "ImageId":
                self.df["ImageId"] = self.df["ImageId"].apply(lambda x: "%s.JPEG" % x)
            elif col == "PredictionString":
                self.df["PredictionString"] = self.df["PredictionString"].str.split(" ", expand=True)[0]
        """
        # create label mapping
        sorted_labels = sorted(self.df.iloc[:, 1].unique())
        self.label_map = {label: idx for idx, label in enumerate(sorted_labels)}

        # print dataset summary
        print(
            "Found %d samples belonging to %d classes in %s dataset."
            % (len(self.df), len(self.df.iloc[:, 1].unique()), folder)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = os.path.join(self.dataset_dir, self.df.iloc[idx, 0])
        label_str = self.df.iloc[idx, 1]
        label = self.label_map[label_str]

        # load image
        image = Image.open(file_name)
        image = self.transform(image)

        # duplicate channels if needed
        if image.shape[0] != 3:
            image = image.expand(3, -1, -1)

        return image, label

    def get_labels(self):
        return [self.label_map[label] for label in self.df.iloc[:, 1]]
    
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


def load_data():
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
    dataset_dir = '/mnt/data/RadImageNet'
    trainset = MyDataset(dataset_dir, "RadImageNet_data", os.path.join(dataset_dir, "RadImageNet_split/RadiologyAI_train.csv"), keep_columns=["filename", "label"], transform=train_transform)
    valset = MyDataset(dataset_dir, "RadImageNet_data", os.path.join(dataset_dir, "RadImageNet_split/RadiologyAI_val.csv"), keep_columns=["filename", "label"], transform=val_transform)

    return trainset, valset

def valTop5(val_loader, model, criterion, top_n=5, device="cpu", verbose=True):
    if verbose:
        iterator = tqdm(val_loader, total=len(val_loader), desc="Validate")
    else:
        iterator = val_loader

    model.eval()  # set the model to evaluation mode

    # iterate over the dataset
    val_loss = 0.0
    topn_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, label in iterator:
            data, label = data.to(device), label.to(device)

            # inference
            pred = model(data)
            loss = criterion(pred, label)
            val_loss += loss.item()

            # get top-n predictions
            _, pred_topn = pred.topk(top_n, 1, largest=True, sorted=True)
            correct = pred_topn.eq(label.view(-1, 1).expand_as(pred_topn))
            topn_correct += correct.sum().item()
            total_samples += data.size(0)

    # average loss for the validation set
    val_loss /= len(val_loader)

    # compute top-n accuracy
    topn_acc = topn_correct / total_samples

    return val_loss, topn_acc
    
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    total_loss = []
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)

        targets = targets.to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()

    scheduler.step()

    epoch_loss = sum(total_loss)/len(total_loss)
    
    return epoch_loss

def pretrain(target_flag):
    seed=42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    with radt.run.RADTBenchmark() as run:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = torchvision.models.resnet18(weights=None, num_classes=165)
        net.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        trainset, valset = load_data()

        train_loader = DataLoader(trainset, batch_size=256, sampler=ImbalancedDatasetSampler(trainset), num_workers=16)
        
        val_loader = DataLoader(valset, batch_size=256, shuffle=False)

        for key, value in vars(args).items():
            run.log_param(key, value)

        print('==> Building and training model...')
        path=f'/mnt/models/doju_pre-trained_for/{target_flag}.pt'
        early_stopping = EarlyStopping(patience=10, verbose=True, path=path)
        for epoch in range(45):  # loop over the dataset multiple times
            print(epoch)
            train_loss = train_epoch(net, train_loader, criterion, optimizer, scheduler, device)
            loss, train_top5_acc = valTop5(train_loader, net, criterion, device=device)
            print('train loss, auc:', train_loss, train_top5_acc)
            val_loss, top5_acc = valTop5(val_loader, net, criterion, device=device)
            print('val loss, top5_acc:', val_loss, top5_acc)
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        run.log_metric("top5_acc", top5_acc)
        run.log_metric("stopped_at", epoch)
        print(top5_acc)
        print("Finished Training")

pretrain(args.target_flag)