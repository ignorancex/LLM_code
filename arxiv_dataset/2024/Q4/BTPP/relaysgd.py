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

from gpu_work import Worker, Tracking_Worker, Relay_Worker
from multi_dataset import get_dataset, get_evaluate_datasets, get_warm_up_datasets
from multi_model import get_model
from topology import get_topology, get_relay_neighbor

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
    for rank in range(args.nodes):
        torch.manual_seed(rank)
        model = deepcopy(model_init)
        model.cuda(args.gpu_rank)
        train_loader = get_dataset(args, rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        schedule = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
        worker = Relay_Worker(rank, model, train_loader, criterion, optimizer, schedule, args)
        worker_list.append(worker)

    """
    prepare the neighborhood and relay message
    """
    for worker in worker_list:
        neighbor_list = get_relay_neighbor(args, worker.rank)
        worker.init_neighbor(neighbor_list)

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
                    for worker in worker_list[1:]:
                        param.data += worker.model.state_dict()[name].data
                    param.data /= args.nodes
                loss_all, acc_all = loss_acc_all(tmp_model, criterion, train_loader_eval, test_loader_eval, args)

                data_dict['train_loss'].append(loss_all)
                data_dict['test_acc'].append(acc_all)
            
            for worker in worker_list:
                worker.calculate_grad()
                worker.update_model()
                worker.copy_model()

            """
            do the Relay-Sum
            """
            message_list = []
            count_list = []
            for worker in worker_list:
                message_list.append(deepcopy(worker.neighbor_relay_message))
                count_list.append(deepcopy(worker.neighbor_relay_count))
            for worker in worker_list:
                for j in worker.neighbor_list:
                    complementary_neighbor = [item for item in worker.neighbor_list if item != j]
                    worker.neighbor_relay_count[j] = 0
                    for name, param in worker.model.named_parameters():
                        worker.neighbor_relay_message[j][name] = torch.zeros_like(param.data)
                    for k in complementary_neighbor:
                        for name, param in worker.model.named_parameters():
                            worker.neighbor_relay_message[j][name] += message_list[k][worker.rank][name]
                        worker.neighbor_relay_count[j] += count_list[k][worker.rank]
                    worker.neighbor_relay_count[j] += 1
                    for name, param in worker.model.named_parameters():
                        worker.neighbor_relay_message[j][name] += param.data
            for worker in worker_list:
                N_count = 0
                for name, param in worker.model.named_parameters():
                    param.data = torch.zeros_like(param.data)
                for j in worker.neighbor_list:
                    for name, param in worker.model.named_parameters():
                        param.data += worker_list[j].neighbor_relay_message[worker.rank][name] 
                    N_count += worker_list[j].neighbor_relay_count[worker.rank]
                N_count += 1
                for name, param in worker.model.named_parameters():
                    param.data += worker.model_copy[name] 
                    param.data /= N_count
                    



    fname = str(args.keep) + 'relaysgd' + args.topo + args.datasets + args.model + str(args.nodes)
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
