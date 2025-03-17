#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import grad

from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent, )
from models import *
from models.aaron import Aaron
from utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch Evaluation")
parser.add_argument("--model", type=str, help="pretrained model")
parser.add_argument("--dataset", type=str, help="dataset to test")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--PGD_steps", type=int, help="steps of PGD attacks")
parser.add_argument("--num_particles",
                    type=int,
                    help="number of ensemble inference")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.dataset == "cifar":
    NET = VGG("VGG16")
    NET = torch.nn.DataParallel(NET)
elif args.dataset == "stl":
    NET = Aaron(10)
    NET = torch.nn.DataParallel(NET)

# Data
print("==> Preparing data..")

if args.dataset == "cifar":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root="/data/datasets/CV/",
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="/data/datasets/CV/",
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

elif args.dataset == "stl":
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.STL10(root="/data/datasets/CV",
                                          split="train",
                                          download=True,
                                          transform=transform_train)
    testset = torchvision.datasets.STL10(root="/data/datasets/CV",
                                         split="test",
                                         download=True,
                                         transform=transform_test)
    classes = (
        "plane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    )

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=8)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=8)

# Model
print("==> Testing model..")

from bayes import BayesWrap

class Object(object):
    pass


opt = Object()
opt.num_particles = args.num_particles
net = BayesWrap(opt, NET)

net = net.to(device)
checkpoint = torch.load(args.model)
state_dict = checkpoint["net"]

# initialize posteriors from the fixed trained particles
net.load_state_dict(state_dict)
net.eval()

criterion = nn.CrossEntropyLoss()

net.eval()
test_budgets = [0, 0.015, 0.035, 0.055, 0.07]


def EOT_PGD(testloader):
    for test_budget in test_budgets:
        total = 0
        correct = 0
        adv_correct = 0
        print("Testing for Epsilon:  {}".format(test_budget))
        for i, (images, labels) in enumerate(testloader):
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            if test_budget == 0:
                # no attack
                outputs = net(images)
                _, predicted = torch.max(outputs.data,
                                         1)  # Prediction on the clean image
                correct += (predicted == labels).sum().item()

                total += labels.size(0)
            else:
                # net = E[theta] achieved by the average of Monte Carlo samples
                steps = args.PGD_steps
                images_adv = projected_gradient_descent(
                    net, images, test_budget, 2 / 255, steps, np.inf)
                adv_output = net(images_adv)

                _, adv_predicted = torch.max(adv_output.data,
                                             1)  # Prediction on the adv images
                adv_correct += (adv_predicted == labels).sum().item()
                total += labels.size(0)
                progress_bar(
                    i,
                    len(testloader),
                    "Acc_adv: %.3f%% (%d/%d)" % (
                        100.0 * adv_correct / total,
                        adv_correct,
                        total,
                    ),
                )

        if test_budget == 0:
            print(
                "Benign accuracy of the model w/0 adverserial attack on test images is : {} ({}/{}) %"
                .format(100 * correct / total, correct, total))
        else:
            print(
                "Accuracy of the model with adverserial attack on test images is : {} %"
                .format(100 * adv_correct / total))


if __name__ == "__main__":
    EOT_PGD(testloader)
