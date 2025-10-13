import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFilter, ImageOps
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
num_epochs = 200


def test(net, testloader, epoch=1):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.0 * correct / total
        print(
            "\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%"
            % (epoch, loss.item(), acc)
        )
    return acc.detach().cpu().numpy()


def test_distributed(net, testloader, local_rank, epoch=1):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    criterion = criterion.cuda(local_rank)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(local_rank, non_blocking=True), targets.cuda(local_rank, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            #torch.distributed.barrier()
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.0 * correct / total
        print(
            "\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%"
            % (epoch, loss.item(), acc)
        )
    return acc.detach().cpu().numpy()


def train(net, trainloader, epoch=1, lr=0.1):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    optimizer = optim.SGD(net.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)

    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=100)
    #niter_per_ep = len(trainloader)
    #schedule = cosine_scheduler(base_value=lr, final_value=5e-4, epochs=100, niter_per_ep=niter_per_ep, warmup_epochs=10)
    print("\n=> Training Epoch #%d, LR=%.4f" % (epoch, learning_rate(lr, epoch)))
    #print("\n=> Training Epoch #%d, LR=%.4f" % (epoch, schedule[(epoch-1)*niter_per_ep]))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #optimizer = optim.SGD(net.parameters(), lr=schedule[(epoch-1)*niter_per_ep + batch_idx], momentum=0.9, weight_decay=5e-4)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        outputs = net(inputs.float())
        loss = criterion(outputs, targets)  # Loss

        #torch.distributed.barrier()
        
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write("\r")
        sys.stdout.write(
            "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%"
            % (
                epoch,
                num_epochs,
                batch_idx + 1,
                len(trainloader),
                loss.item(),
                100.0 * correct / total,
            )
        )
        sys.stdout.flush()


def train_distributed(net, trainloader, local_rank, epoch=1, lr=0.1):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    criterion = criterion.cuda(local_rank)
    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4
    )

    print("\n=> Training Epoch #%d, LR=%.4f" % (epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(local_rank, non_blocking=True), targets.cuda(local_rank, non_blocking=True)  # GPU settings
        optimizer.zero_grad()
        outputs = net(inputs.float())
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write("\r")
        sys.stdout.write(
            "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%"
            % (
                epoch,
                num_epochs,
                batch_idx + 1,
                len(trainloader),
                loss.item(),
                100.0 * correct / total,
            )
        )
        sys.stdout.flush()


def learning_rate(init, epoch):
    optim_factor = 0
    if epoch > 150:
        optim_factor = 3
    elif epoch > 100:
        optim_factor = 2
    elif epoch > 50:
        optim_factor = 1

    return init * math.pow(0.2, optim_factor)


def cosine_scheduler(base_value=5e-4, final_value=1e-6, epochs=100, niter_per_ep=1000, warmup_epochs=10, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def save_list(file_name, path):
    file = open(path, 'w')
    for fp in file_name:
        file.write(str(fp))
        file.write('\n')
    file.close()


def load_list(path):
    data = []
    file_handler =open(path, mode='r')
    contents = file_handler.readlines()
    for name in contents:
        name = name.strip('\n')
        #data.append(float(name))
        data.append(name)
    return data


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
