import sys 
sys.path.append("..") 
import models

import torch
from torch.utils.data import DataLoader
from dataset import mnist_test_data
from params import model_path


def get_accuracy(model, dataloader: DataLoader):
    total = 0
    correct = 0
    for input_data, gt_labels in dataloader:
        input_data, gt_labels = input_data.cuda(), gt_labels.cuda()
        _, predicted = torch.max(model(input_data), 1)
        total += len(gt_labels)
        correct_labels = torch.eq(predicted, gt_labels)
        correct += correct_labels.sum().item()
    return correct / total

net_file = 'lenet_0_bs_log_0'
net = torch.load(f'{model_path}{net_file}.t7')['net']
net.cuda()
dataloader_test = DataLoader(mnist_test_data, 4)
acc = get_accuracy(net, dataloader_test)
print(acc)
