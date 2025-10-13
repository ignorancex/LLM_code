from models import HugeModel, HugeModelMultiTask
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from losses import return_task_loss
from copy import deepcopy
from torch.optim import lr_scheduler
from abc import ABC, abstractmethod

class AbstractClient(ABC):

    def __init__(self, dataset, global_model, device, args, network) -> None:
        super().__init__()
        self.dataset = dataset
        self.dest_task = args.task
        self.model = deepcopy(global_model).to(device)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.init_lr)
        self.criterion = return_task_loss(self.dest_task)
        self.sgd_per_epoch = args.sgd_per_epoch
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_decay_rate)
        self.device = device
        self.criterion = return_task_loss(self.dest_task)
        self.network = network

    def sync_params(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
    
    @abstractmethod
    def train_once(self):
        
        raise NotImplementedError("Method not implemented")

class FedAvgClient(AbstractClient):

    def __init__(self, dataset, global_model, device, args, network):
        super(FedAvgClient, self).__init__(dataset, global_model, device, args, network)
        self.no_improvement_tolerance = args.no_improvement_tolerance

    def train_once(self):

        loss_hist = []
        best_loss = np.Inf
        no_improvement = 0
        steps_taken = 0

        self.model.train()
        while steps_taken < self.sgd_per_epoch:
            for idx, (image, label) in enumerate(self.dataloader):
                if steps_taken >= self.sgd_per_epoch:
                    break
                steps_taken += 1

                image = image.to(self.device).float()
                if self.dest_task == "class_scene":
                    label = label[self.dest_task].to(self.device)
                else:
                    label = label[self.dest_task].to(self.device).float()

                _, outputs = self.model(image)
                loss = self.criterion(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_hist.append(loss.item())

                if len(loss_hist) >= 5 and np.mean(loss_hist[-5:]) < best_loss:
                    best_loss = np.mean(loss_hist[-5:])
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= self.no_improvement_tolerance:
                    self.scheduler.step()

class FedProxClient(AbstractClient):

    def __init__(self, dataset, global_model, device, args, network):
        super(FedProxClient, self).__init__(dataset, global_model, device, args, network)
        self.no_improvement_tolerance = args.no_improvement_tolerance
        self.mu = args.mu
        

    def train_once(self):
        
        self.loss_hist = []
        self.best_loss = np.Inf
        self.no_improvement = 0

        self.model.train()
        for idx, (image, label) in enumerate(self.dataloader):
            if idx >= self.sgd_per_epoch:
                break

            image = image.to(self.device).float()
            if self.dest_task == "class_scene":
                label = label[self.dest_task].to(self.device)
            else:
                label = label[self.dest_task].to(self.device).float()

            _, outputs = self.model(image)
            loss = self.criterion(outputs, label)
            loss += 0.5 * self.mu * sum([torch.norm(local_param - global_param) ** 2 for local_param, global_param in zip(self.model.parameters(), self.network.global_model.parameters())])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_hist.append(loss.item())

            if np.mean(self.loss_hist[-5:]) < self.best_loss:
                self.best_loss = np.mean(self.loss_hist[-5:])
                self.no_improvement = 0
            else:
                self.no_improvement += 1
            
            if self.no_improvement >= self.no_improvement_tolerance:
                self.scheduler.step()



class MultiTaskClient():

    def __init__(self, dataset, clip_model, device, args) -> None:
        self.dataset = dataset
        self.dest_tasks = self.dataset.dest_tasks
        self.device = device
        self.model = HugeModelMultiTask(deepcopy(clip_model), self.dest_tasks).to(self.device)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.init_lr)
        self.sgd_per_epoch = args.sgd_per_epoch
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_decay_rate)
        self.criteria = {return_task_loss(task) for task in self.dest_tasks}
    
    def sync_params(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

    def train_once(self):
        self.model.train()
        for idx, (image, labels) in enumerate(self.dataloader):
            
            if idx >= self.sgd_per_epoch:
                break

            self.optimizer.zero_grad()
            image = image.to(self.device).float()
            _, output = self.model(image)
            total_loss = 0
            for task in self.dest_tasks:
                label = labels[task].to(self.device).float()
                loss = self.criteria[task](output, label)
                total_loss += loss
            total_loss.backward()
            self.optimizer.step()

        self.scheduler.step()

