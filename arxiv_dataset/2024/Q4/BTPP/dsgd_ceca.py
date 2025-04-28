from config import get_config
import warnings
warnings.simplefilter('ignore')

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from copy import deepcopy
import collections
from scipy.io import savemat

from gpu_work import Worker, Ceca_Worker
from multi_dataset import get_dataset, get_evaluate_datasets, get_warm_up_datasets
from multi_model import get_model
from topology import get_topology

import math

import numpy as np
import random

def work():
    args = get_config()
    run(args)

def run(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    train_loader_eval, test_loader_eval = get_evaluate_datasets(args)

    """
    prepare the topology
    """
    matrix_Ws, matrix_Wgs = get_topology(args)
    criterion = nn.CrossEntropyLoss()
    worker_list = []

    """
    preparation for workers
    """
    torch.manual_seed(0)
    model_init = get_model(args)
    model_init.cuda(args.gpu_rank)
    if args.warm_up > 0:
        optimizer = torch.optim.SGD(model_init.parameters(), lr=args.warm_up_lr)
        train_loader = get_warm_up_datasets(args)
        for ite in range(args.warm_up):
            model_init.train()
            data, target = next(iter(train_loader))
            data, target = data.cuda(args.gpu_rank), target.cuda(args.gpu_rank)
            output = model_init(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """
    here I add n virtual nodes to better clarify the algorithm
    """
    for rank in range(args.nodes):
        torch.manual_seed(rank)
        model = deepcopy(model_init)
        model.cuda(args.gpu_rank)
        train_loader = get_dataset(args, rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        schedule = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
        worker = Ceca_Worker(rank, model, train_loader, criterion, optimizer, schedule, args)
        worker_list.append(worker)

    for rank in range(args.nodes):
        torch.manual_seed(rank + args.nodes)
        model = deepcopy(model_init)
        model.cuda(args.gpu_rank)
        train_loader = get_dataset(args, rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        schedule = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
        worker = Ceca_Worker(rank, model, train_loader, criterion, optimizer, schedule, args)
        worker_list.append(worker)

    tau = int(math.ceil(math.log(args.nodes, 2.0)))
    bi = bin(args.nodes - 1)

    """
    start the training
    """
    data_dict = collections.defaultdict(list)
    with tqdm(total=args.iterations) as t:
        for ite in range(args.iterations):
            t.update(1)
            """
            calculate the loss and accuracy
            tmp_model: global average model or weighted average model
            """
            if ite % args.record == 0:
                tmp_model = deepcopy(worker_list[0].model)
                for name, param in tmp_model.named_parameters():
                    for worker in worker_list[1:args.nodes]:
                        param.data += worker.model.state_dict()[name].data
                    param.data /= args.nodes
                loss_all, acc_all = loss_acc_all(tmp_model, criterion, train_loader_eval, test_loader_eval, args)

                data_dict['train_loss'].append(loss_all)
                data_dict['test_acc'].append(acc_all)

            tt = ite % tau
            matrix_W = torch.from_numpy(matrix_Ws[tt]).cuda(args.gpu_rank)
            matrix_Wg = torch.from_numpy(matrix_Wgs[tt]).cuda(args.gpu_rank)
            dig = int(bi[2 + tt])

            if dig == 1:
                for worker in worker_list[0:args.nodes]:
                    worker.calculate_grad()
                    worker.copy_grad()
                    worker.copy_model()
            else:
                for worker in worker_list[args.nodes:2*args.nodes]:
                    worker.calculate_grad()
                    worker.copy_grad()
                    worker.copy_model()

            for worker in worker_list:
                for name, param in worker.model.named_parameters():
                    param.grad = torch.zeros_like(worker.grad_copy[name].data)
                    param.data = torch.zeros_like(worker.grad_copy[name].data)

            for worker in worker_list:
                for name, param in worker.model.named_parameters():
                    for i in range(2*args.nodes):
                        param.grad.data += matrix_Wg[worker.rank][i] * worker_list[i].grad_copy[name].data
                        param.data += matrix_W[worker.rank][i] * worker_list[i].model_copy[name].data

            for worker in worker_list:
                worker.update_model()
                        

    fname = str(args.keep) + 'dsgd_ceca' + args.topo + args.datasets + args.model + str(args.nodes)
    savemat(file_name = './record/' + fname + ".mat", mdict = data_dict)

def loss_acc_all(model, criterion, train_loader, test_loader, args):
    """
    calculate the train loss and the test accuracy
    """
    model.cuda(args.gpu_rank)
    model.eval()
    total, correct, ites, total_loss = 0, 0, 0, 0
    for batch in test_loader:
        data, target = batch[0].to(args.gpu_rank), batch[1].to(args.gpu_rank)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total += len(target)
        correct += p.eq(target).sum().item()
    acc = correct / total

    for batch in train_loader:
        ites += 1
        data, target = batch[0].to(args.gpu_rank), batch[1].to(args.gpu_rank)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
    total_loss = total_loss / ites

    return total_loss, acc     


if __name__ == '__main__':
    work()
