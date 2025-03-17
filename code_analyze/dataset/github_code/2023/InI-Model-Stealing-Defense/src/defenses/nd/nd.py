import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from datasets import REGISTRY_INFO, REGISTRY_TRAIN_DATASET, REGISTRY_TEST_DATASET
from models import get_family, REGISTRY_MODEL

class ND:
    def __init__(self, model, dataset, dataset_root, exp_path, batch_size=256, num_workers=0, epochs=50, lr=0.1, step_size=50, lr_gamma=0.1, device=torch.device("cuda"), **kwargs):
        self.device = device
        self.exp_path = exp_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.step_size = step_size
        self.lr_gamma = lr_gamma

        self.dataset_train = REGISTRY_TRAIN_DATASET[dataset](root=dataset_root)
        self.dataset_test = REGISTRY_TEST_DATASET[dataset](root=dataset_root)

        self.model = REGISTRY_MODEL[get_family(dataset)][model](REGISTRY_INFO[dataset][0]).to(self.device)


    def __call__(self, x):
        return self.model(x)
    
    def eval(self):
        self.model.eval()

    def load(self, path):
        self.model.load_state_dict(torch.load(path + "/model.pth", map_location=self.device))

    def train(self):
        criterion = nn.CrossEntropyLoss()

        opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        sch = optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=self.lr_gamma)

        dataloader_train = DataLoader(self.dataset_train, self.batch_size, shuffle=True, num_workers=self.num_workers)
        dataloader_test = DataLoader(self.dataset_test, self.batch_size, shuffle=False, num_workers=self.num_workers)

        best_acc = 0
        for epoch in range(1, self.epochs + 1):
            s = time.time()
            train_loss, train_acc = self.train_epoch(dataloader_train, opt, criterion)
            test_acc = self.test(dataloader_test)
            if sch:
                sch.step()
            e = time.time()
            time_epoch = e - s
            print(
                "Epoch: {} train_loss: {:.3f} train_acc: {:.2f}%, test_acc: {:.2f}% time: {:.1f}".format(
                    epoch, train_loss, train_acc * 100, test_acc * 100, time_epoch
                )
            )
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), f"{self.exp_path}/model_best.pth")
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"{self.exp_path}/model_{epoch}.pth")

    def train_epoch(self, data_loader, opt, criterion):
        """
        Train for 1 epoch
        """
        self.model.train()
        running_loss = correct = 0.0
        n_batches = len(data_loader)
        for (x, y) in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            pred = self.model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pred_class = torch.argmax(pred, dim=-1)
            if y.ndim == 2:
                y = torch.argmax(y, dim=-1)
            correct += (pred_class == y).sum().item()

        loss = running_loss / n_batches
        acc = correct / len(data_loader.dataset)
        return loss, acc

    def test(self, data_loader):
        self.model.eval()
        correct = 0.0
        with torch.no_grad():
            for (x, y) in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.__call__(x)
                pred_class = torch.argmax(pred, dim=1)
                correct += (pred_class == y).sum().item()
            acc = correct / len(data_loader.dataset)
        return acc