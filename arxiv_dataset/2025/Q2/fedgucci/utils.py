from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.backends import cudnn
import math
from pyhessian import hessian
from torch.optim import Optimizer
# TODO update this
from models_dict import resnet, cnn, mlp, vgg, resnet_old, gnn_lstm, roberta, vit, resnet_cifar
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from sam import SAM, SAMFED
from typing import List, Tuple, Union
import socket
import platform
from sklearn.metrics import matthews_corrcoef


##############################################################################
# Tools
##############################################################################

  
def calculate_mean_std(dictionary): 
    '''
    input: dict = {seed0:list, seed1:list, seed2:list}
    output: mean_list, std_list, string-like mean ± std
    '''
    list_length = len(dictionary[list(dictionary.keys())[0]])  
  
    mean_list = []  
    std_list = []  
    str_mean_std = []
  
    for i in range(list_length):  
        values = [dictionary[key][i] for key in dictionary]  

        mean_value = np.mean(values)  
        std_value = np.std(values)  

        mean_list.append(mean_value)  
        std_list.append(std_value)
        str_mean_std.append("{:.3g}".format(mean_value)+'±'+"{:.3g}".format(std_value))  
  
    return mean_list, std_list, str_mean_std


def calculate_barrier_and_stats(linear_loss_recorder, interpolate_loss_recorder, linear_acc_recorder, interpolate_acc_recorder):

    # Calculate the avg and std of the interpolate_loss
    interpolate_loss_recorder_avg = {}
    interpolate_loss_recorder_std = {}
    for method in list(interpolate_loss_recorder.keys()):
        interpolate_loss_recorder_avg[method], interpolate_loss_recorder_std[method], _ = calculate_mean_std(interpolate_loss_recorder[method])

    interpolate_acc_recorder_avg = {}
    interpolate_acc_recorder_std = {}
    for method in list(interpolate_acc_recorder.keys()):
        interpolate_acc_recorder_avg[method], interpolate_acc_recorder_std[method], _  = calculate_mean_std(interpolate_acc_recorder[method])
    
    landscape_recorder = {'loss':{'avg':interpolate_loss_recorder_avg, 'std':interpolate_loss_recorder_std},
                           'acc':{'avg':interpolate_acc_recorder_avg, 'std':interpolate_acc_recorder_std}}
    
    # Calculate the avg acc value and the interpolate_middle value
    avg_acc_of_two_models = {}
    interpolate_acc_of_two_models = {}
    for method in list(interpolate_acc_recorder.keys()):
        avg_acc_of_two_models[method] = {}
        interpolate_acc_of_two_models[method] = {}
        for seed in list(interpolate_acc_recorder[method].keys()):
            avg_acc_of_two_models[method][seed] = [(interpolate_acc_recorder[method][seed][0] + interpolate_acc_recorder[method][seed][-1])/2]
            interpolate_acc_of_two_models[method][seed] = [interpolate_acc_recorder[method][seed][int(len(interpolate_acc_recorder[method][seed])//2)]]

        _, _, interpolate_acc_of_two_models[method] = calculate_mean_std(interpolate_acc_of_two_models[method])
        _, _, avg_acc_of_two_models[method] = calculate_mean_std(avg_acc_of_two_models[method])

    
    # Calculate the barrier ± std
    acc_barrier_recorder = {}
    for method in list(interpolate_acc_recorder.keys()):
        acc_barrier_recorder[method] = {}
        for seed in list(interpolate_acc_recorder[method].keys()):
            barrier_list =  [a - b for a, b in zip(linear_acc_recorder[method][seed], interpolate_acc_recorder[method][seed])]
            max_barrier = max(barrier_list)
            max_index = barrier_list.index(max_barrier)
            acc_barrier_recorder[method][seed] = [max_barrier / linear_acc_recorder[method][seed][max_index]]

        _, _, acc_barrier_recorder[method] = calculate_mean_std(acc_barrier_recorder[method])

    loss_barrier_recorder = {}
    for method in list(interpolate_loss_recorder.keys()):
        loss_barrier_recorder[method] = {}
        for seed in list(interpolate_loss_recorder[method].keys()):
            barrier_list =  [a - b for a, b in zip(interpolate_loss_recorder[method][seed], linear_loss_recorder[method][seed])]
            max_barrier = max(barrier_list)
            loss_barrier_recorder[method][seed] = [max_barrier]

        _, _, loss_barrier_recorder[method] = calculate_mean_std(loss_barrier_recorder[method])
    
    barriers_recorder = {'acc_barrier':acc_barrier_recorder, 'loss_barrier':loss_barrier_recorder,
                         'avg_acc':avg_acc_of_two_models, 'interpolate_acc':interpolate_acc_of_two_models}

    return barriers_recorder, landscape_recorder


def calculate_loss_barrier_and_stats(linear_loss_recorder, interpolate_loss_recorder):

    # Calculate the avg and std of the interpolate_loss
    interpolate_loss_recorder_avg = {}
    interpolate_loss_recorder_std = {}
    for method in list(interpolate_loss_recorder.keys()):
        interpolate_loss_recorder_avg[method], interpolate_loss_recorder_std[method], _ = calculate_mean_std(interpolate_loss_recorder[method])

    
    landscape_recorder = {'loss':{'avg':interpolate_loss_recorder_avg, 'std':interpolate_loss_recorder_std}}
    

    loss_barrier_recorder = {}
    for method in list(interpolate_loss_recorder.keys()):
        loss_barrier_recorder[method] = {}
        for seed in list(interpolate_loss_recorder[method].keys()):
            barrier_list =  [a - b for a, b in zip(interpolate_loss_recorder[method][seed], linear_loss_recorder[method][seed])]
            max_barrier = max(barrier_list)
            loss_barrier_recorder[method][seed] = [max_barrier]

        _, _, loss_barrier_recorder[method] = calculate_mean_std(loss_barrier_recorder[method])
    
    barriers_recorder = {'loss_barrier':loss_barrier_recorder}

    return barriers_recorder, landscape_recorder

def merge_two_params(parameters0, parameters1, alpha0, alpha1):
    merged_params = copy.deepcopy(parameters0)
    for name_param in parameters0:
        merged_params[name_param] = alpha0*parameters0[name_param] + alpha1*parameters1[name_param]
    return merged_params


def two_nodes_landscape(alpha_list, y_label, loss_recorder, dataset, model, path):
    plt.figure(figsize=(12, 7))
    b = plt.subplot()
    b.set_xlabel('alpha')
    b.set_ylabel(y_label)
    b.set_title('Linear Mode Connectivity (' + str(dataset)+', '+str(model)+ ')')

    i = 0
    for method in loss_recorder:
        plt.plot(alpha_list, loss_recorder[method], color = sns.color_palette(n_colors=5)[i], linewidth=1, label=method) # 0.85
        i += 1

    plt.legend()

    plt.savefig(path, bbox_inches='tight')
    

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)
    
    def clear(self):
        self.steps = 0
        self.total = 0

def softmax_fuct(lrs):
    '''
    lrs is dict as {0:3, 1:3, 2:4}
    '''
    exp_cache = []
    softmax_lrs = {}
    for i in range(len(lrs)):
        exp_cache.append(math.exp(lrs[i]))
    
    for i in range(len(lrs)):
        softmax_lrs[i] = exp_cache[i]/sum(exp_cache)
    
    return softmax_lrs

def cos(x, y):
    fuct = nn.CosineSimilarity(dim=0)
    result = fuct(x, y)
    result = result.detach().cpu().numpy().tolist()
    return result

def get_cosGrad_matrix(gradients):
    client_num = len(gradients)
    matrix = [[0.0 for _ in range(client_num)] for _ in range(client_num)]

    for i in range(client_num):
        for j in range(client_num):
            if matrix[j][i] != 0.0:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = cos(gradients[i], gradients[j])
    
    return matrix

def model_parameter_vector(args, model):
    if 'fedawo' in args.server_method:
        vector = model.flat_w
    else:
        param = [p.view(-1) for p in model.parameters()]
        vector = torch.cat(param, dim=0)
    return vector

##############################################################################
# Initialization function
##############################################################################

# TODO update this model initializer
def init_model(model_type, args, num_classes):
    original_num_classes = num_classes
    if args.dataset in ('cifar10', 'fmnist'):
        num_classes = 10
    else:
        num_classes = 100
    if original_num_classes is not None:
        num_classes = original_num_classes


    if model_type == 'CNN':
        if args.dataset == 'cifar10':
            if args.client_method == 'fedrod':
                model = cnn.CNNCifar10ForFedrod()
            else:
                model = cnn.CNNCifar10()
        else:
            model = cnn.CNNCifar100()
    elif model_type == 'CNN_FEDSAM':
        model = cnn.CNNCifar10FedSam()
    elif model_type == 'ResNet20':
        model = resnet_cifar.resnet20_cifar(num_classes)
    elif model_type == 'NewResNet20':
        model = resnet_cifar.resnet20_cifar(num_classes)

    elif model_type == 'ResNet18':
        model = resnet_old.ResNet18()
    elif model_type == 'ResNet34':
        model = resnet_old.ResNet34()

    elif model_type == 'ResNet56':
        model = resnet.ResNet56(num_classes)
    elif model_type == 'ResNet110':
        model = resnet.ResNet110(num_classes)
    elif model_type == 'WRN56_2':
        model = resnet.WRN56_2(num_classes)
    elif model_type == 'WRN56_4':
        model = resnet.WRN56_4(num_classes)
    elif model_type == 'WRN56_8':
        model = resnet.WRN56_8(num_classes)

    elif model_type == 'MLP_h1_w200':
        model = mlp.MLP_h1_w200(args.dataset)
    elif model_type == 'MLP_h2_w200':
        model = mlp.MLP_h2_w200(args.dataset)
    elif model_type == 'MLP_h3_w200':
        model = mlp.MLP_h3_w200(args.dataset)
    elif model_type == 'MLP_h4_w200':
        model = mlp.MLP_h4_w200(args.dataset)
    elif model_type == 'MLP_h5_w200':
        model = mlp.MLP_h5_w200(args.dataset)
    elif model_type == 'MLP_h6_w200':
        model = mlp.MLP_h6_w200(args.dataset)
    elif model_type == 'MLP_h8_w200':
        model = mlp.MLP_h8_w200(args.dataset)
    elif model_type == 'MLP_h2_w400':
        model = mlp.MLP_h2_w400(args.dataset)
    elif model_type == 'MLP_h2_w800':
        model = mlp.MLP_h2_w800(args.dataset)
    elif model_type == 'MLP_h2_w1600':
        model = mlp.MLP_h2_w1600(args.dataset)
    elif model_type == 'MLP_h2_w5':
        model = mlp.MLP_h2_w5(args.dataset)
    elif model_type == 'MLP_h2_w10':
        model = mlp.MLP_h2_w10(args.dataset)
    elif model_type == 'MLP_h2_w25':
        model = mlp.MLP_h2_w25(args.dataset)
    elif model_type == 'MLP_h2_w50':
        model = mlp.MLP_h2_w50(args.dataset)
    elif model_type == 'MLP_h2_w100':
        model = mlp.MLP_h2_w100(args.dataset)
    elif model_type == 'MLP':
        model = mlp.MLP()
    
    elif model_type == 'VGG11':
        model = vgg.VGG('VGG11', num_classes = num_classes)
    elif model_type == 'VGG13':
        model = vgg.VGG('VGG13', num_classes = num_classes)
    elif model_type == 'VGG16':
        model = vgg.VGG('VGG16', num_classes = num_classes)
    elif model_type == 'VGG19':
        model = vgg.VGG('VGG19', num_classes = num_classes)

    elif model_type == 'GCNN':
        model = gnn_lstm.GCNN()
    elif model_type == 'LSTM':
        model = gnn_lstm.LSTM()
    
    elif model_type == 'LeNet':
        model = cnn.LeNet5()
    
    elif model_type == 'roberta-base':
        model = roberta.RobertaBase(model_path=args.model_path, num_labels=num_classes, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    elif model_type == 'Pretrained_ResNet18':
        model = resnet.Pretrained_ResNet18(num_classes)
    
    elif model_type == 'ViT':
        model = vit.ViT(num_classes=num_classes)
    
    else:
        raise NotImplementedError


    return model

def init_optimizer(num_id, model, args, is_sam: bool = False):
    optimizer = []
    if num_id == -1:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.local_wd_rate)
    elif num_id > -1 and 'fedprox' in args.client_method:
        optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
    
    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    os.environ["PYTHONSEED"] = str(seed)

##############################################################################
# Training function
##############################################################################

def generate_matchlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr

def load_data(args, cluster_id, X_batch, Y_batch):
    # swap label for the robustness experiment
    X_batch2 = X_batch
    Y_batch2 = Y_batch

    if args.noniid_type == 'swap':
        Y_batch2 = Y_batch.numpy().tolist()
        if args.num_cluster == 4:
            # 4 clusters
            if args.corrupt_percent == 1:
                if cluster_id == 3:
                    for x in range(len(Y_batch2)):
                        if Y_batch2[x] == 0:
                            Y_batch2[x] = 1
                        elif Y_batch2[x] == 1:
                            Y_batch2[x] = 0
            elif args.corrupt_percent == 2:
                if cluster_id == 2:
                    for x in range(len(Y_batch2)):
                        if Y_batch2[x] == 0:
                            Y_batch2[x] = 1
                        elif Y_batch2[x] == 1:
                            Y_batch2[x] = 0
                elif cluster_id == 3:
                    for x in range(len(Y_batch2)):
                        if Y_batch2[x] == 2:
                            Y_batch2[x] = 3
                        elif Y_batch2[x] == 3:
                            Y_batch2[x] = 2
            elif args.corrupt_percent == 3:
                if cluster_id == 1:
                    for x in range(len(Y_batch2)):
                        if Y_batch2[x] == 4:
                            Y_batch2[x] = 5
                        elif Y_batch2[x] == 5:
                            Y_batch2[x] = 4
                elif cluster_id == 2:
                    for x in range(len(Y_batch2)):
                        if Y_batch2[x] == 0:
                            Y_batch2[x] = 1
                        elif Y_batch2[x] == 1:
                            Y_batch2[x] = 0
                elif cluster_id == 3:
                    for x in range(len(Y_batch2)):
                        if Y_batch2[x] == 2:
                            Y_batch2[x] = 3
                        elif Y_batch2[x] == 3:
                            Y_batch2[x] = 2
        Y_batch2 = torch.Tensor(Y_batch2).long()

    return X_batch2, Y_batch2


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # g = g.cuda()
                if p.grad is None:
                    continue
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])

    @torch.no_grad()
    def step_with_mask(self, global_params, mask):
        for group in self.param_groups:
            for idx, (p, g) in enumerate(zip(group['params'], global_params)):
                # g = g.cuda()
                d_p = p.grad.data + group['mu'] * (p.data - g.data)

                p_size = d_p.size()
                p_grad = d_p.reshape(-1)
                
                p_mask = mask[idx]

                p_grad = p_grad * p_mask
                d_p = p_grad.view(p_size).to(d_p.dtype)

                p.data.add_(d_p, alpha=-group['lr'])


##############################################################################
# Validation function
##############################################################################

def validate(args, node, which_dataset = 'validate'):
    node.model.cuda().eval() 
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = load_data(args, node.cluster_id, data, target)
            data, target = cuda(data), target.cuda()
            if args.client_method == 'fedrod':
                feature, logit, out = node.model(data, return_hidden_state=True)
                if which_dataset == 'validate':
                    logit_p = node.p_head(out)
                    output = logit + logit_p
                else:
                    output = logit
            else:
                output = node.model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    
    if 'cola' in args.dataset:
        print("compute mcc")
        # get preds and targets
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = load_data(args, node.cluster_id, data, target)
                data, target = cuda(data), target.cuda()
                output = node.model(data)
                y_preds.extend(output.argmax(dim=1).cpu())
                y_trues.extend(target.cpu())
        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)
        
        # compute the matthews correlation coefficient
        mcc = matthews_corrcoef(y_true=y_trues, y_pred=y_preds)
        return mcc
                    
    return acc

def testloss(args, node, which_dataset = 'validate'):
    node.model.cuda().eval()  
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = load_data(args, node.cluster_id, data, target)
            data, target = data.cuda(), target.cuda()
            output = node.model(data)
            loss_local =  F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss)/len(loss)
    return loss_value

# Functions for FedAWO with param as an input
def validate_with_param(args, node, param, which_dataset = 'validate'):
    node.model.cuda().eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = load_data(args, node.cluster_id, data, target)
            data, target = data.cuda(), target.cuda()
            output = node.model.forward_with_param(data, param)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc

def testloss_with_param(args, node, param, which_dataset = 'validate'):
    node.model.cuda().eval()  
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = load_data(args, node.cluster_id, data, target)
            data, target = data.cuda(), target.cuda()
            output = node.model.forward_with_param(data, param)
            loss_local =  F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss)/len(loss)
    return loss_value

def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        List of parameters [, names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters

def get_local_ip():
    try:
      local_ip = os.environ["SSH_CONNECTION"].split()[2]
      return local_ip
    except Exception:
        return "无法获取IP地址"

def cuda(data):
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i].cuda()
    else:
        data = data.cuda()
    return data

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
    def __call__(self, val_metric):
        if self.best_metric == None:
            self.best_metric = val_metric
        elif self.best_metric - val_metric < self.min_delta:
            self.best_metric = val_metric
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_metric - val_metric > self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True