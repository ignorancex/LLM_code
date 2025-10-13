import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from util import *
import resnet
import prune as P
import argparse
import random
import time
import pickle
import numpy as np

from dataloader import get_cifar_10_loaders

def get_model():
    if args.model_name == 'resnet_152':
        model = resnet.resnet152()
        args.checkpoint = 'resnet_pruning/models/ResNet_152_10.th'
        model = nn.DataParallel(model, device_ids=args.gpu)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        model.eval()
        
    return model

def prune_model_layer(model, original_model, train_loader, val_loader):
    accs = prune(model, original_model, train_loader, val_loader)
    return accs

def prune(model, original_model, train_loader, val_loader):
    accs = []

    layers = [model.module.layer1, model.module.layer2, model.module.layer3]
    modules = len(layers)

    l, acc_best = validate(val_loader, model, L_cls_f)
    print('acc_best', acc_best)
   
    for layer_index in range(modules):

        blocks = len(layers[layer_index])
        for block_index in range(blocks):
            print('module.layer' + str(layer_index + 1) + '.' + str(block_index))

            layers[layer_index][block_index] = P.prune(model, train_loader, layers[layer_index][block_index], args.pruning_ratio, args.pruning_method)

            model = model.cuda()

            #loss, acc = validate(val_loader, model, L_cls_f)

            #print("[Pre Loss: {:.3f}. Acc: {:2.2f}%.".format(loss, acc))

            model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
                
            fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)
                
            loss, acc = validate(val_loader, model, L_cls_f)

            print("[Post Loss: {:.3f}. Acc: {:2.2f}%.".format(loss, acc))

            accs.append(acc)

    return accs

def distillation_loss(y_logit, t_logit, T=2):
    return F.kl_div(F.log_softmax(y_logit / T, 1), F.softmax(t_logit / T, 1), reduction='sum') / y_logit.size(0)

def fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt, use_distill=True):
    global args
    model.train()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        model_opt.zero_grad()
        z = model(input)
        z_ori = original_model(input)
        L = L_cls_f(z, target)
        if use_distill:
           L += distillation_loss(z, z_ori)
        L.backward()

        model_opt.step()
    model.eval()

def validate(val_loader, model, L_cls_f):
    global args

    loss = 0
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            z = model(input)
            L_cls = L_cls_f(z, target)
            loss += L_cls.item()

            _, predicted = torch.max(z.data, 1)
            total += input.size(0)
            correct += (predicted == target).sum().item()

    #print('== {} Loss : {:.5f}. Acc : {:2.2f}%'.format(prefix, loss / len(val_loader), correct / total * 100))

    return loss / len(val_loader), correct / total * 100

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=128)
#parser.add_argument('--sample_level', type=float, default=0.02)
#parser.add_argument('--sampler', type=str, default='TPE')
parser.add_argument('--ft_lr', type=float, default=0.001)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--pruning_ratio', type=float, default=0)
#parser.add_argument('--threshold', type=float, default=0)
#parser.add_argument('--freeze_percentile', type=float, default=0)
parser.add_argument('--search_trails', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_name', type=str, default='resnet_152')
parser.add_argument('--pruning_method', type=str, default='l1')

global args, iters
args = parser.parse_args()

print('Ratio: ' + str(args.pruning_ratio))

set_seed(args.seed)

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

args.gpu = [int(i) for i in args.gpu.split(',')]

L_cls_f = nn.CrossEntropyLoss().cuda()

def ice_axe_tester(model, original_model, train_loader, val_loader):

    accs = prune_model_layer(model, original_model, train_loader, val_loader)

    print(accs)

    return accs[-1]

time_start = time.time()
print('using', args.pruning_method)

train_loader, val_loader = get_cifar_10_loaders(1, args.batch_size)

model = get_model()
_, acc_ori = validate(val_loader, model, L_cls_f)
print("Ori Acc: {:2.2f}%.".format(acc_ori))

original_model = copy.deepcopy(model)

acc =  ice_axe_tester(model, original_model, train_loader, val_loader)

time_total = time.time() - time_start
print('time_total', time_total)

res = {}
res['time'] = time_total
res['acc'] = acc
print(res)

with open('/project/experiment/' + args.model_name + '_' + args.pruning_method + '_seed_' + str(args.seed) + '_ratio_' + str(args.pruning_ratio * 100) + '.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
