import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import copy
from util import *
import resnet
from mobilenetv2 import MobileNetV2
import prune as P
import argparse
import random
import pickle
import numpy as np

def get_weight_norm(model):
    weight_norm = {}
    layers = [model.module.layer1, model.module.layer2, model.module.layer3]
    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            name = 'module.layer' + str(layer_index + 1) + '.' + str(block_index) + '.' + 'conv1.weight'
            weight_norm[name] = torch.norm(layers[layer_index][block_index].conv1.weight).item()
            
            name = 'module.layer'+str(layer_index + 1)+'.'+str(block_index)+'.'+'conv2.weight'
            weight_norm[name] = torch.norm(layers[layer_index][block_index].conv2.weight).item()
    return weight_norm        

def get_weight_norm_mobile(model):
    weight_norm = {}
    layers = model.module.layers
    for layer_index in range(len(layers)):
        name = 'module.layer' + str(layer_index + 1) + '.' + 'conv1.weight'
        weight_norm[name] = torch.norm(layers[layer_index].conv1.weight)

        name = 'module.layer' + str(layer_index + 1) + '.' + 'conv2.weight'
        weight_norm[name] = torch.norm(layers[layer_index].conv2.weight)

        name = 'module.layer' + str(layer_index + 1) + '.' + 'conv3.weight'
        weight_norm[name] = torch.norm(layers[layer_index].conv3.weight)
    return weight_norm

def prune(model, ratio, model_name):
    trace = {}
    l, acc_best = validate(val_loader, model, L_cls_f, '* ')
    
    if model_name == 'resnet_152':
        layers = [model.module.layer1, model.module.layer2, model.module.layer3]
        modules = len(layers)
        for layer_index in range(modules):
            blocks = len(layers[layer_index])
            for block_index in range(blocks):

                print('module.layer' + str(layer_index + 1) + '.' + str(block_index))
                trace['module.layer' + str(layer_index + 1) + '.' + str(block_index)] = {}

                layers[layer_index][block_index] = P.prune_layer_by_layer_resnet(layers[layer_index][block_index], ratio, False)

                model = model.cuda()

                weight = {}
                get_weight(model, weight)
                trace['module.layer' + str(layer_index + 1) + '.' + str(block_index)]['pre'] = copy.deepcopy(weight)
            
                model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
                fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)

                weight = {}
                get_weight(model, weight)
                trace['module.layer' + str(layer_index + 1) + '.' + str(block_index)]['post'] = copy.deepcopy(weight)
    
    elif model_name == 'mobilenetv2':
        layers = model.module.layers
        modules = len(layers)
        print(modules) 
        for layer_index in range(modules):
            print('module.layer' + str(layer_index + 1))

            trace['module.layer' + str(layer_index + 1)] = {}

            layers[layer_index] = P.prune_layer_by_layer_mobile(layers[layer_index], ratio, False)

            model = model.cuda()

            weight = {}
            get_weight_mobile(model, weight)
            trace['module.layer' + str(layer_index + 1)]['pre'] = copy.deepcopy(weight)

            model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
            fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)

            weight = {}
            get_weight_mobile(model, weight)
            trace['module.layer' + str(layer_index + 1)]['post'] = copy.deepcopy(weight)


    loss, acc = validate(val_loader, model, L_cls_f, '* ')
    print("[Pre Loss: {:.3f}. Acc: {:2.2f}%.".format(loss, acc))
    return trace

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

def validate(val_loader, model, L_cls_f, prefix='', print=False):
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

    if print: print('== {} Loss : {:.5f}. Acc : {:2.2f}%'.format(prefix, loss / len(val_loader), correct / total * 100))

    return loss / len(val_loader), correct / total * 100

def get_params_zero_non(model):
    layer_names = get_layer_names(model)
    num_param_zero_non = 0
    for i in range(1, len(layer_names)):
        index = len(layer_names) - i
        layer_name = layer_names[index].split('.')[:-1]
        layer = model
        for i in range(len(layer_name) - 1):
            layer = layer._modules[layer_name[i]]

        num_param_zero_non+=torch.count_nonzero(layer._modules[layer_name[-1]].weight)

    return num_param_zero_non

def get_params_zero_non(model):
    layer_names = get_layer_names(model)
    num_param_zero_non = 0
    for i in range(1, len(layer_names)):
        index = len(layer_names) - i
        layer_name = layer_names[index].split('.')[:-1]
        layer = model
        for i in range(len(layer_name) - 1):
            layer = layer._modules[layer_name[i]]

        num_param_zero_non+=torch.count_nonzero(layer._modules[layer_name[-1]].weight)

    return num_param_zero_non

def get_weight(model, weight):
    layers = [model.module.layer1, model.module.layer2, model.module.layer3]
    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            name = 'module.layer' + str(layer_index + 1) + '.' + str(block_index) + '.' + 'conv1.weight'
            weight[name] = layers[layer_index][block_index].conv1.weight

            name = 'module.layer'+str(layer_index + 1)+'.'+str(block_index)+'.'+'conv2.weight'
            weight[name] = layers[layer_index][block_index].conv2.weight


def get_weight_mobile(model, weight):
    layers = model.module.layers
    for layer_index in range(len(layers)):
        name = 'module.layer' + str(layer_index + 1) + '.' + 'conv1.weight'
        weight[name] = layers[layer_index].conv1.weight

        name = 'module.layer'+str(layer_index + 1) + '.'+'conv2.weight'
        weight[name] = layers[layer_index].conv2.weight

        name = 'module.layer' + str(layer_index + 1) + '.' + 'conv3.weight'
        weight[name] = layers[layer_index].conv3.weight

def processe_trace(trace):
    final_trace = {}
    for name in trace.keys():
        all = []
        for layer in trace[name]['pre'].keys():
            a_norm_pre = torch.sum(torch.abs(trace[name]['pre'][layer].view(-1)), dim=0)

            a_norm_post = torch.sum(torch.abs(trace[name]['post'][layer].view(-1)), dim=0)
            w_change = torch.abs(a_norm_post - a_norm_pre)
            all.append({layer: (w_change/w_norm[layer]).item()})
        final_trace[name] = all
    return final_trace

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
parser.add_argument('--ft_lr', type=float, default=0.001)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--model_name', type=str, default='resnet152')
parser.add_argument('--ratio_pruning', type=float, default=0)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

print('Model: ' + str(args.model_name))
print('Ratio: ' + str(args.ratio_pruning * 100))

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

set_seed(args.seed)

args.gpu = [int(i) for i in args.gpu.split(',')]
L_cls_f = nn.CrossEntropyLoss().cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
        transforms.ToTensor(), normalize]), download=True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(), normalize])),
    batch_size=128, shuffle=False,
    num_workers=args.workers, pin_memory=True)

device = 'cuda'
if args.model_name == 'resnet_152':
    model = resnet.resnet152()
    args.checkpoint = 'smaller_scale/models/ResNet_152_10.th'
elif args.model_name == 'mobilenetv2':
    model = MobileNetV2()
    args.checkpoint = 'smaller_scale/models/mobilev2.th'

model = nn.DataParallel(model, device_ids=args.gpu).cuda()
checkpoint = torch.load(args.checkpoint, map_location='cuda:1')

model.load_state_dict(checkpoint['state_dict'])

model.eval()
original_model = copy.deepcopy(model)
if args.model_name == 'resnet_152':
    w_norm = get_weight_norm(model)
elif args.model_name == 'mobilenetv2':
    w_norm = get_weight_norm_mobile(model)

total_param = get_params_zero_non(original_model)

loss, acc = validate(val_loader, model, L_cls_f, '* ')
print("[Init {:02d}] Loss: {:.3f}. Acc: {:2.2f}%.".format(1, loss, acc))
    
trace = prune(model, args.ratio_pruning, args.model_name)

final_trace = processe_trace(trace)
    
res = {}
res['trace']= final_trace
    
with open('/project/experiment/' + args.model_name + '_' + str(args.ratio_pruning) + '_seed_' + str(args.seed) + '.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
current = get_params_zero_non(model)
print('current paras _ out: ' + str(current))
print('ratio:' + str(current.item() / total_param.item()))

