import torch

from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn as nn

from copy import deepcopy

class Worker:
    """
    Worker for DSGD
    """
    def __init__(self, rank, model:Module, 
                train_loader:DataLoader, criterion,
                optimizer, schedule, args):
        self.rank = rank
        self.model = model
        self.train_loader = train_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.schedule = schedule

        self.model.cuda(args.gpu_rank)
        self.criterion.cuda(args.gpu_rank)

        self.args = args

    def calculate_grad(self):
        self.model.train()
        data, target, _ = next(iter(self.train_loader))
        data, target = data.cuda(self.args.gpu_rank), target.cuda(self.args.gpu_rank)
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

    def update_model(self):
        self.optimizer.step()
        self.schedule.step()


    def copy_model(self):
        self.model_copy = deepcopy(self.model.state_dict())

    

class Tracking_Worker(Worker):
    """
    Worker for Gradient Tracking
    """
    def __init__(self, rank, model: Module, train_loader: DataLoader, criterion, optimizer, schedule, args):

        super().__init__(rank, model, train_loader, criterion, optimizer, schedule, args)

        self.grad_tracking = deepcopy(self.model.state_dict())
        self.grad_copy = deepcopy(self.model.state_dict())

    def copy_grad(self):
        for name, param in self.model.named_parameters():
            self.grad_copy[name].data = param.grad.data

class Ceca_Worker(Worker):
    """
    Worker for DSGD-CECA 1p/2p
    """
    def __init__(self, rank, model: Module, train_loader: DataLoader, criterion, optimizer, schedule, args):

        super().__init__(rank, model, train_loader, criterion, optimizer, schedule, args)

        self.grad_copy = deepcopy(self.model.state_dict())
        self.model_copy = deepcopy(self.model.state_dict())

    def copy_grad(self):
        for name, param in self.model.named_parameters():
            self.grad_copy[name].data = param.grad.data

    def copy_model(self):
        self.model_copy = deepcopy(self.model.state_dict())



class Relay_Worker(Worker):
    """
    Worker for RelaySGD
    """
    def __init__(self, rank, model: Module, train_loader: DataLoader, criterion, optimizer, schedule, args):

        super().__init__(rank, model, train_loader, criterion, optimizer, schedule, args)

        self.grad_copy = deepcopy(self.model.state_dict())
        self.model_copy = deepcopy(self.model.state_dict())

    def init_neighbor(self, neighbor_list):
        self.neighbor_list = neighbor_list
        self.neighbor_relay_message = []
        self.neighbor_relay_count = []
        for i in range(self.args.nodes):
            self.neighbor_relay_count.append(0)
            if i in self.neighbor_list:
                self.neighbor_relay_message.append(deepcopy(self.model.state_dict()))
            else:
                self.neighbor_relay_message.append(0)
        for i in range(self.args.nodes):
            if i in self.neighbor_list:
                for name, param in self.model.named_parameters():
                    self.neighbor_relay_message[i][name] = torch.zeros_like(param.data)


    def copy_grad(self):
        for name, param in self.model.named_parameters():
            self.grad_copy[name].data = param.grad.data

    def copy_model(self):
        self.model_copy = deepcopy(self.model.state_dict())


class D2_Worker(Worker):
    """
    Worker for D2
    """
    def __init__(self, rank, model: Module, train_loader: DataLoader, criterion, optimizer, schedule, args):

        super().__init__(rank, model, train_loader, criterion, optimizer, schedule, args)

        self.model_copy = deepcopy(self.model.state_dict())
        self.grad_copy = deepcopy(self.model.state_dict())
        self.grad_pre = deepcopy(self.model.state_dict())
        self.model_pre = deepcopy(self.model.state_dict())
        self.model_pre_copy = deepcopy(self.model.state_dict())


    def refresh_model(self):
        self.model_pre_copy = deepcopy(self.model_pre)
        self.model_pre = deepcopy(self.model.state_dict())

    def refresh_grad(self):
        self.grad_pre = deepcopy(self.grad_copy)
        for name, param in self.model.named_parameters():
            self.grad_copy[name].data = param.grad.data

    def copy_model(self):
        self.model_copy = deepcopy(self.model.state_dict())


        