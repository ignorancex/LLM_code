import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from util import *
import prune_dense as P
import argparse
import random
import time
import pickle
import numpy as np

from dataloader import get_tiny_imagenet_loaders
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def get_model():
    if args.model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 200)
        args.checkpoint = 'densenet_pruning/models/dense_tinyimagenet_10_epochs.pth'
        model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
    return model

def prune_model_layer(model, original_model, train_loader, val_loader, train_loader_large, val_loader_full):
    accs = prune(model, original_model, train_loader, val_loader, train_loader_large, val_loader_full)
    return accs

def prune(model, original_model, train_loader, val_loader, train_loader_large, val_loader_full):
    accs = []
    
    layers = model.module.features
    l, acc_best = validate(val_loader, model, L_cls_f)
    print('acc_best', acc_best)
    
    for layer_index in range(len(layers)):
        if layers[layer_index].__class__.__name__ != '_DenseBlock':
            continue
            
        for block_index, block in enumerate(layers[layer_index]):

            print('module.layer' + str(layer_index + 1) + '.' + str(block_index))

            layers[layer_index][block] = P.prune(model, train_loader, layers[layer_index][block], args.pruning_ratio, args.pruning_method)

            model = model.cuda()

            model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)

            fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)

            if layer_index > 6:
                fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)

            loss, acc = validate(val_loader, model, L_cls_f)

            print("[Post Loss: {:.3f}. Acc: {:2.2f}%.".format(loss, acc))

            accs.append(acc)

    best_acc = 0
    model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
        
    for e in range(args.final_epochs):
        fine_tuning(model, original_model, train_loader_large, val_loader_full, L_cls_f, model_opt)
        loss, acc = validate(val_loader, model, L_cls_f)
        if acc > best_acc:
            best_acc = acc
        print("Loss: {:.3f}. Acc: {:2.2f}%.".format(loss, acc))
    accs.append(best_acc)
    return accs

def distillation_loss(y_logit, t_logit, T=2):
    return F.kl_div(F.log_softmax(y_logit / T, 1), F.softmax(t_logit / T, 1), reduction='sum') / y_logit.size(0)

def fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt, use_distill=True):
    global args
    model.train()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        model_opt.zero_grad()
        with autocast():
            z = model(input)
            #z_ori = original_model(input)
        L = L_cls_f(z, target)
        #if use_distill:
        #    L += distillation_loss(z, z_ori)
        #L.backward()
        scaler.scale(L).backward()
        scaler.step(model_opt)
        scaler.update()
        #model_opt.step()
    model.eval()

def validate(val_loader, model, L_cls_f):
    global args

    loss = 0
    model.eval()

    total = 0
    correct = 0
    
    with autocast():
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
            
                target = target.cuda(non_blocking=True)
                z = model(input)
                L_cls = L_cls_f(z, target)
                loss += L_cls.item()

                _, predicted = torch.max(z.data, 1)
                total += input.size(0)
                correct += (predicted == target).sum().item()

    #print('== Loss : {:.5f}. Acc : {:2.2f}%'.format(prefix, loss / len(val_loader), correct / total * 100))

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
#parser.add_argument('--sample_level_train', type=float, default=0.002)
#parser.add_argument('--sample_level_test', type=float, default=0.002)
#parser.add_argument('--sampler', type=str, default='TPE')
parser.add_argument('--ft_lr', type=float, default=0.005)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--pruning_ratio', type=float, default=0)
#parser.add_argument('--threshold', type=float, default=0)
#parser.add_argument('--freeze_percentile', type=float, default=0)
#parser.add_argument('--exp_rounds', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_name', type=str, default='densenet')
parser.add_argument('--pruning_method', type=str, default='l1')
parser.add_argument('--search_rounds', type=int, default=3)
parser.add_argument('--final_epochs', type=int, default=4)

global args, iters
args = parser.parse_args()

print('Ratio: ' + str(args.pruning_ratio))

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

set_seed(args.seed)

args.gpu = [int(i) for i in args.gpu.split(',')]
L_cls_f = nn.CrossEntropyLoss().cuda()

def ice_axe_tester(model, original_model, train_loader, val_loader, train_loader_large, val_loader_full):

    accs = prune_model_layer(model, original_model, train_loader, val_loader, train_loader_large, val_loader_full)

    print(accs)

    return accs[-1]

time_start = time.time()

print('using', args.pruning_method)

train_loader, val_loader = get_tiny_imagenet_loaders(0.02, 0.02, args.batch_size) #0.02 0.02 0.1 0.1

train_large, val_full = get_tiny_imagenet_loaders(1, 1, args.batch_size)

model = get_model()
_, acc_ori = validate(val_loader, model, L_cls_f)
print("Ori Acc: {:2.2f}%.".format(acc_ori))

original_model = copy.deepcopy(model)
acc =  ice_axe_tester(model, original_model, train_loader, val_loader, train_large, val_full)

time_total = time.time() - time_start
print('time_total', time_total)

res = {}
res['time'] = time_total
res['acc'] = acc
print(res)

with open('/project/experiment/' + args.model_name  + '_' + args.pruning_method + '_seed_' + str(args.seed) + '_ratio_' + str(args.pruning_ratio * 100) + '.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
