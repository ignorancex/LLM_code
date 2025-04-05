
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse
import copy

from src.dataloader import make_imagenette_loader, make_data_loader
from src.trainers import eval_model, train_baseline
from src.wrappers import Dummy
from torchvision import models

parser = argparse.ArgumentParser()


parser.add_argument('-ep', '--epochs', type=int, default=20)
parser.add_argument('-wd', '--weight_decay', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-lp', '--log_path', type=str, default='/tmp/')
parser.add_argument('-nv', '--nesterov', action='store_true')

parser.add_argument('-ar', '--arch', type=str,
                    choices=['vgg16'], default='vgg16')

parser.add_argument('-mp', '--model_path', type=str,
                    default='models/baseline/cifar10/vgg16.pth',
                    help='Path to model weights')

parser.add_argument('-ds', '--dataset', type=str, default='imagenette',
                    choices=['cifar10', 'cifar100', 'food101', 'imagenette', 'imagewoof'],
                    help='Dataset for training')


args = parser.parse_args()
epochs = args.epochs
lr = args.learning_rate
bs = args.batch_size
arch = args.arch
log_path = args.log_path
model_path = args.model_path
dataset = args.dataset
freq = 5
weight_decay = args.weight_decay
nesterov = args.nesterov


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataloaders, dataset_sizes = make_imagenette_loader(bs, path='./data/imagenette2-320/')
dataloaders, dataset_sizes = make_data_loader(bs, dataset=dataset)

model = models.vgg16_bn(pretrained=False)
model.classifier[-1] = nn.Linear(in_features=4096, out_features=100)
model = Dummy(model) # returns empty activations because I'm too lazy
                     # to write another training loop

model = model.to(device)

criterion = nn.CrossEntropyLoss()

model_path = 'vgg16_{}_lr_{}_wd_{}_nesterov_{}_epochs_{}'.format(
                                                dataset, lr, weight_decay,
                                                nesterov, epochs)

model_id = model_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = './models/' + model_path


optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                         weight_decay=weight_decay, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,
                                            milestones=[
                                                np.ceil(epochs*.33),
                                                np.ceil(epochs*.66)
                                            ],
                                            gamma=.1)


if not os.path.exists(save_path):
    os.makedirs(save_path)

writer = SummaryWriter(log_dir=log_path + model_id)

best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(0, epochs):
    train_baseline(
            model, criterion, dataloaders,
            dataset_sizes, optimizer_ft, exp_lr_scheduler,
            writer, epoch, phase='train')

    epoch_acc = eval_model(
                    model, criterion, dataloaders,
                    dataset_sizes, writer, epoch=epoch)

    if epoch_acc > best_acc:
        best_model_wts = copy.deepcopy(model.state_dict())


model.load_state_dict(best_model_wts)

final_acc = eval_model(
                model, criterion, dataloaders,
                dataset_sizes, writer, epoch=epochs)
writer.close()
torch.save(model.state_dict(), './models/' + model_id + '.pt')
