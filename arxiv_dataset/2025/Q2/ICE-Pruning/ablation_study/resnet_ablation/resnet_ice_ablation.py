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
import optuna

from dataloader import get_cifar_10_loaders

def get_model():
    if args.model_name == 'resnet_152':
        model = resnet.resnet152()
        args.checkpoint = 'resnet_ablation/models/ResNet_152_10.th'
        model = nn.DataParallel(model, device_ids=args.gpu)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        model.eval()
        
    return model

def get_weight_norm():
    model = get_model()
    weight_norm = {}
    layers = [model.module.layer1, model.module.layer2, model.module.layer3]
    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            name = 'module.layer' + str(layer_index + 1) + '.' + str(block_index) + '.' + 'conv1.weight'
            weight_norm[name] = torch.norm(layers[layer_index][block_index].conv1.weight).item()

            name = 'module.layer'+str(layer_index + 1)+'.'+str(block_index)+'.'+'conv2.weight'
            weight_norm[name] = torch.norm(layers[layer_index][block_index].conv2.weight).item()
    return weight_norm

def freeze_model(model, original_model, train_loader, val_loader, fr):
    trace = prune([], model, original_model, train_loader, val_loader, 1, 0,  0.001, 0, 0, 0, True)
    config = get_freezed_layers(fr, trace)
    model = get_model()
    layers = [model.module.layer1, model.module.layer2, model.module.layer3]
    for layer_index in range(len(layers)):
        blocks = len(layers[layer_index])
        for block_index in range(blocks):
            freeze_layer(config, layers[layer_index][block_index], layer_index, block_index)
    return model, config
    
def freeze_layer(config, layer, l_index, b_index):
    name1 = 'module.layer' + str(l_index + 1) + '.' + str(b_index) + '.' + 'conv1.weight'
    name2 = 'module.layer' + str(l_index + 1) + '.' + str(b_index) + '.' + 'conv2.weight'
    if name1 in config:
        for param in layer.conv1.parameters():
            param.requires_grad = False
        for param in layer.bn1.parameters():
            param.requires_grad = False
    if name2 in config:
        for param in layer.conv2.parameters():
            param.requires_grad = False
        for param in layer.bn2.parameters():
            param.requires_grad = False

def prune_model_layer(freeze_config, model, original_model, train_loader, val_loader, thre, base, p, beta, delta, auto_search=False):
    accs = prune(freeze_config, model, original_model, train_loader, val_loader, 2, thre, base, p, beta, delta, auto_search)
    return accs

def prune(freeze_config, model, original_model, train_loader, val_loader, depth, thre, base, p, beta, delta, auto_search=False):
    trace = {}
    accs = []

    layers = [model.module.layer1, model.module.layer2, model.module.layer3]
    modules = len(layers)

    if depth == 1:
        modules = 1

    else:
        l, acc_best = validate(val_loader, model, L_cls_f)
        print('acc_best', acc_best)
        print('thre', thre)

    for layer_index in range(modules):
        if depth == 1:
            blocks = 1
        else:
            blocks = len(layers[layer_index])
        for block_index in range(blocks):
            if not auto_search:
                print('module.layer' + str(layer_index + 1) + '.' + str(block_index))

            if depth == 1:
                trace['module.layer' + str(layer_index + 1) + '.' + str(block_index)] = {}

            layers[layer_index][block_index] = P.prune(model, train_loader, layers[layer_index][block_index], args.pruning_ratio, args.pruning_method)

            if not args.no_freeze:
                freeze_layer(freeze_config, layers[layer_index][block_index], layer_index, block_index)
            model = model.cuda()
        
            if depth == 1:
                weight = {}
                get_weight(model, weight)
                trace['module.layer' + str(layer_index + 1) + '.' + str(block_index)]['pre'] = copy.deepcopy(weight)

            loss, acc = validate(val_loader, model, L_cls_f)
            
            if not auto_search:
                print("[Pre Loss: {:.3f}. Acc: {:2.2f}%.".format(loss, acc))

            if args.no_threshold or depth ==1 or acc_best - acc >= thre:
                amount = get_params_zero_non(model) / get_params_zero_non(original_model)
                if depth ==1 or args.no_adapt_lr:
                    learn_rate = base
                else:
                    learn_rate = cal_lr(amount.item(), base, p, beta, delta) #0.001, 0.12, 50, 0.00075

                model_opt = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=1e-4)
                
                fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)
                
                loss, acc = validate(val_loader, model, L_cls_f)
                if not auto_search:
                    print("[Post Loss: {:.3f}. Acc: {:2.2f}%.".format(loss, acc))

            if depth == 1:
                weight = {}
                get_weight(model, weight)
                trace['module.layer' + str(layer_index + 1) + '.' + str(block_index)]['post'] = copy.deepcopy(weight)

            accs.append(acc)

        if depth == 1:
            return trace

    return accs

def cal_lr(x, base, p, beta, delta):
    return base - (delta/(1 + pow((x / ((1 - p)*2 - x)), beta)))

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

def get_weight(model, weight):
    layers = [model.module.layer1, model.module.layer2, model.module.layer3]
    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            name = 'module.layer' + str(layer_index + 1) + '.' + str(block_index) + '.' + 'conv1.weight'
            weight[name] = layers[layer_index][block_index].conv1.weight

            name = 'module.layer'+str(layer_index + 1)+'.'+str(block_index)+'.'+'conv2.weight'
            weight[name] = layers[layer_index][block_index].conv2.weight

def get_freezed_layers(percentile, trace):
    all = []
    for name in trace.keys():
        for layer in trace[name]['pre'].keys():
            a_norm_pre = torch.sum(torch.abs(trace[name]['pre'][layer].view(-1)), dim=0)

            a_norm_post = torch.sum(torch.abs(trace[name]['post'][layer].view(-1)), dim=0)
            
            all.append((layer, (torch.abs(a_norm_post - a_norm_pre)/weight_norm[layer]).item()))
    
    all.sort(key=lambda x: (x[1]))
    length = len(all)
    #print(all)
    freeze_layers = all[:math.ceil(percentile*length)]

    config = []
    for layer in freeze_layers:
        config.append(layer[0])

    print(config)
    return config

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
parser.add_argument('--sample_level', type=float, default=0.02)
#parser.add_argument('--sampler', type=str, default='TPE')
#parser.add_argument('--ft_lr', type=float, default=0.001)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--pruning_ratio', type=float, default=0)
#parser.add_argument('--threshold', type=float, default=0)
#parser.add_argument('--freeze_percentile', type=float, default=0)
parser.add_argument('--search_trails', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_name', type=str, default='resnet_152')
parser.add_argument('--pruning_method', type=str, default='l1')
parser.add_argument('--no_threshold', action='store_true')
parser.add_argument('--no_freeze', action='store_true')
parser.add_argument('--no_adapt_lr', action='store_true')

global args, iters
args = parser.parse_args()

print('Ratio: ' + str(args.pruning_ratio))

set_seed(args.seed)

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

args.gpu = [int(i) for i in args.gpu.split(',')]

L_cls_f = nn.CrossEntropyLoss().cuda()

def ice_axe_tester(model, original_model, train_loader, val_loader, thre, base, p, beta, delta, fr=0, auto_search=False):
    #model.eval()
    if not args.no_freeze:
        model_freezed, freeze_config = freeze_model(model, original_model, train_loader, val_loader, fr)
    else:
        model_freezed = model
        freeze_config = []

    loss, acc = validate(val_loader, model, L_cls_f)
    print("[Init {:02d}] Loss: {:.3f}. Acc: {:2.2f}%.".format(1, loss, acc))

    start_time = time.time()
    print('start at: ' + str(start_time))

    accs = prune_model_layer(freeze_config, model_freezed, original_model, train_loader, val_loader, thre, base, p, beta, delta, auto_search)

    total_time = time.time() - start_time
    print('start at: ' + str(start_time))
    print('end at: ' + str(time.time()))
    print('Total time: {} seconds'.format(total_time))

    print(accs)

    return total_time, accs[-1]

train_loader, val_loader = get_cifar_10_loaders(args.sample_level, args.batch_size)

def objective(trial):
    thre = trial.suggest_float('thre', 0.5, 1.5)
    fr = trial.suggest_float('fr', 0.3, 0.9)
    base = trial.suggest_float('base', 0.001, 0.005)
    p = trial.suggest_float('p', 0.1, 0.2)
    beta = trial.suggest_float('beta', 40, 60)
    delta = trial.suggest_float('delta', 0.0006, 0.0008)
    
    model = get_model()
    _, acc_ori = validate(val_loader, model, L_cls_f)
    print("Acc: {:2.2f}%.".format(acc_ori))

    original_model = copy.deepcopy(model)

    time, acc =  ice_axe_tester(model, original_model, train_loader, val_loader, thre, base, p, beta, delta, fr, True)

    delta_acc = acc_ori - acc
    
    # Normolize
    time_n = time/max(delta_acc, time)
    delta_acc_n = delta_acc/max(delta_acc, time)
    
    return time_n + delta_acc_n

auto_ice_time_start = time.time()
#print('using sampler', args.sampler)
weight_norm = get_weight_norm()

print('using', args.pruning_method)
search_space = {'thre': [0.5, 1.5],'fr':[0.3, 0.9], 'base':[0.001, 0.005], 'p':[0.1, 0.15, 0.2], 'beta':[40, 50, 60], 'delta':[0.0006, 0.0007, 0.0008]}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space, seed=args.seed))
study.optimize(objective, n_trials=args.search_trails)

best_para = study.best_params
print('best_para', best_para)

train_loader, val_loader = get_cifar_10_loaders(1, args.batch_size)

model = get_model()
_, acc_ori = validate(val_loader, model, L_cls_f)
print("Ori Acc: {:2.2f}%.".format(acc_ori))

original_model = copy.deepcopy(model)

_, acc =  ice_axe_tester(model, original_model, train_loader, val_loader, best_para['thre'], best_para['base'], best_para['p'], best_para['beta'], best_para['delta'], best_para['fr'])

auto_ice_time_total = time.time() - auto_ice_time_start
print('auto_ice_time_total', auto_ice_time_total)

res = {}
res['time'] = auto_ice_time_total
res['acc'] = acc
print(res)

ablation_type = ''
if args.no_threshold:
    ablation_type = '_no_threshold'
elif args.no_freeze:
    ablation_type = '_no_freeze'
elif args.no_adapt_lr:
    ablation_type = '_no_adapt_lr'

with open('/project/experiment/' + args.model_name + '_' + args.pruning_method + '_seed_' + str(args.seed) + '_ratio_' + str(args.pruning_ratio * 100) + ablation_type  + '.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
