import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import MSCMRsegDatasetTrain, MSCMRsegDatasetPredict
from model import my_net
import os

NAME = 'C0'
batch_size = 32
lr = 0.01
num_epochs = 40
device = 'cuda'


def eval(model, dataloader):
    total = 0
    correct = 0
    for data in dataloader:
        inputs = data[0].to(device).float()
        labels = data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


def train():
    dataset_train = MSCMRsegDatasetTrain(NAME, 'train')
    dataset_val = MSCMRsegDatasetTrain(NAME, 'val')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    net = my_net()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_batches = len(dataloader_train)
    best_acc = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for data in dataloader_train:
            inputs = data[0].to(device).float()
            labels = data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = eval(net, dataloader_val)

        if acc > best_acc:
            torch.save(net.state_dict(), f'./data/checkpoints/{NAME}/model_best.pt')
            best_acc = acc
        torch.save(net.state_dict(), f'./data/checkpoints/{NAME}/model_latest.pt')

        print(f'Epoch{epoch}: {total_loss / num_batches}, {correct / total}, {acc}')


if __name__ == '__main__':
    train()

# net.load_state_dict(torch.load(f'./data/checkpoints/{NAME}/model_best.pt'))
#
# dataset_train = MSCMRsegDatasetPredict(NAME, 'train')
#
# dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False)
#
# total = 0
# correct = 0
# for data in dataloader:
#     inputs = data[0].to(device).squeeze()
#     label = data[1].to(device)
#     outputs = net(inputs)
#     _, predicted = torch.max(outputs, 1)
#     predicted = torch.mode(predicted).values.item()
#     if predicted == label:
#         correct += 1
#     total += 1
# print(correct / total)
