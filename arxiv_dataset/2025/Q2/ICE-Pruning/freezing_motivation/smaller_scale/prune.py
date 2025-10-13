import math

import torch
import torch.nn as nn

def prune_resnet(net, ratio, independentflag):
    # init
    residue = None
    arg_index = 0
    layers = [net.module.layer1, net.module.layer2, net.module.layer3]

    removed_maps = {'out': {}, 'in': {}}
    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            # identify channels to remove
            layers_size = layers[layer_index][block_index].conv1.weight.shape[0]
            prune_channels = math.ceil(ratio*layers_size)
            remove_channels = [i for i in range(prune_channels)]#channels_index(layers[layer_index][block_index].conv1.weight.data, prune_channels, residue, independentflag)
            #print(prune_layers[arg_index], remove_channels)
            # prune this layer's filter in dim=0
            layers[layer_index][block_index].conv1 = get_new_conv(layers[layer_index][block_index].conv1, remove_channels, 0)
            # prune next layer's filter in dim=1
            layers[layer_index][block_index].conv2, residue = get_new_conv(layers[layer_index][block_index].conv2, remove_channels, 1)
            residue = 0
            # prune bn
            layers[layer_index][block_index].bn1 = get_new_norm(layers[layer_index][block_index].bn1, remove_channels)
            arg_index += 1
            #removed_maps[layer_index][block_index] = list(set([i for i in range(layers_size)]) - set(remove_channels))
            removed_maps['out']['module.layer'+str(layer_index + 1)+'.'+str(block_index)+'.'+'conv1.weight'] = list(set([i for i in range(layers_size)]) - set(remove_channels))
            removed_maps['in']['module.layer'+str(layer_index + 1)+'.'+str(block_index)+'.'+'conv2.weight'] = list(set([i for i in range(layers_size)]) - set(remove_channels))
    net = net.cuda()
    return net, removed_maps


def prune_layer_by_layer_resnet_delta(block, ratio, independentflag, delta, name1, name2):
    # identify channels to remove
    layers_size = block.conv1.weight.shape[0]
    prune_channels = math.ceil(ratio*layers_size)
    remove_channels = channels_index(block.conv1.weight.data, prune_channels, None, independentflag)
    #print(prune_layers[arg_index], remove_channels)
    # prune this layer's filter in dim=0
    block.conv1, new_delta_conv1 = get_new_conv_delta(block.conv1, remove_channels, 0, delta[name1], delta[name2])
    # prune next layer's filter in dim=1
    block.conv2, residue, new_delta_conv2 = get_new_conv_delta(block.conv2, remove_channels, 1, delta[name1], delta[name2])
    residue = 0
    # prune bn
    block.bn1 = get_new_norm(block.bn1, remove_channels)

    delta[name1] = new_delta_conv1
    delta[name2] = new_delta_conv2

    return block

def prune_layer_by_layer_resnet(block, ratio, independentflag):
    # identify channels to remove
    layers_size = block.conv1.weight.shape[0]
    prune_channels = math.ceil(ratio*layers_size)
    remove_channels = channels_index(block.conv1.weight.data, prune_channels, None, independentflag)
    #print(prune_layers[arg_index], remove_channels)
    # prune this layer's filter in dim=0
    block.conv1 = get_new_conv(block.conv1, remove_channels, 0)
    # prune next layer's filter in dim=1
    block.conv2, residue = get_new_conv(block.conv2, remove_channels, 1)
    residue = 0
    # prune bn
    block.bn1 = get_new_norm(block.bn1, remove_channels)

    return block

def prune_layer_by_layer_mobile(block, ratio, independentflag):
    # identify channels to remove
    #print(block.conv2.weight.shape[0], block.conv2.weight.shape[1])
    layers_size = block.conv1.weight.shape[0]

    prune_channels = math.ceil(ratio*layers_size)
    final_groups = layers_size - prune_channels
    remove_channels = channels_index(block.conv1.weight.data, prune_channels, None, independentflag)

    #print(prune_layers[arg_index], remove_channels)
    # prune this layer's filter in dim=0
    block.conv1 = get_new_conv(block.conv1, remove_channels, 0)
    # prune next layer's filter in dim=1

    block.conv2 = get_mobile_conv(block.conv2, remove_channels, 0)
    # prune next layer's filter in dim=1
    block.conv3, residue = get_new_conv(block.conv3, remove_channels, 1)

    # prune bn
    block.bn1 = get_new_norm(block.bn1, remove_channels)
    block.bn2 = get_new_norm(block.bn2, remove_channels)

    block.conv2.groups = final_groups

    return block

def prune_layer_by_layer_wrn(block, ratio, independentflag):
    # identify channels to remove
    layers_size = block.conv1.weight.shape[0]
    prune_channels = math.ceil(ratio * layers_size)
    remove_channels = channels_index(block.conv1.weight.data, prune_channels, None, independentflag)
    # print(prune_layers[arg_index], remove_channels)
    # prune this layer's filter in dim=0
    block.conv1 = get_new_conv(block.conv1, remove_channels, 0)
    # prune next layer's filter in dim=1
    block.conv2, residue = get_new_conv(block.conv2, remove_channels, 1)
    residue = 0
    # prune bn
    block.bn2 = get_new_norm(block.bn2, remove_channels)

    '''
    if len(block.shortcut) == 1:
        layers_size = block.shortcut[0].weight.shape[1]
        prune_channels = math.ceil(ratio * layers_size)
        remove_channels = channels_index(block.shortcut[0].weight.data, prune_channels, None, independentflag)

        # prune next layer's filter in dim=1
        block.shortcut[0], residue = get_new_conv(block.shortcut[0], remove_channels, 1)
    '''
    return block


def get_removing_channels(net, ratio):
    # init
    residue = None
    arg_index = 0
    layers = [net.module.layer1, net.module.layer2, net.module.layer3]

    removed_maps = {'out': {}, 'in': {}}
    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            # identify channels to remove
            layers_size = layers[layer_index][block_index].conv1.weight.shape[0]
            prune_channels = math.ceil(ratio * layers_size)
            remove_channels = channels_index(layers[layer_index][block_index].conv1.weight.data, prune_channels,
                                             residue, False)

            removed_maps['out']['module.layer' + str(layer_index + 1) + '.' + str(block_index) + '.' + 'conv1.weight'] = remove_channels
            removed_maps['in']['module.layer' + str(layer_index + 1) + '.' + str(block_index) + '.' + 'conv2.weight'] = remove_channels

    return removed_maps

def channels_index(weight_matrix, prune_num, residue, independentflag):
    abs_sum = torch.sum(torch.abs(weight_matrix.view(weight_matrix.size(0), -1)), dim=1)
    if independentflag and residue is not None:
        abs_sum = abs_sum + torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    _, indices = torch.sort(abs_sum)
    return indices[:prune_num].tolist()


def select_channels(weight_matrix, remove_channels, dim):
    indices = torch.tensor(list(set(range(weight_matrix.shape[dim])) - set(remove_channels)))
    new = torch.index_select(weight_matrix, dim, indices.cuda())
    if dim == 1:
        residue = torch.index_select(weight_matrix, dim, torch.tensor(remove_channels).cuda())
        return new, residue
    return new


def get_new_conv_delta(old_conv, remove_channels, dim, delta_conv1, delta_conv2):
    if dim == 0:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels,
                             out_channels=old_conv.out_channels - len(remove_channels),
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data = select_channels(old_conv.weight.data, remove_channels, dim)
        new_delta_conv1 = select_channels(delta_conv1, remove_channels, dim)

        if old_conv.bias is not None:
            new_conv.bias.data = select_channels(old_conv.bias.data, remove_channels, dim)
        return new_conv, new_delta_conv1
    else:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels - len(remove_channels), out_channels=old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data, residue = select_channels(old_conv.weight.data, remove_channels, dim)
        new_delta_conv2, residue = select_channels(delta_conv2, remove_channels, dim)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
        return new_conv, residue, new_delta_conv2

def get_new_conv(old_conv, remove_channels, dim):
    if dim == 0:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels,
                             out_channels=old_conv.out_channels - len(remove_channels),
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data = select_channels(old_conv.weight.data, remove_channels, dim)

        if old_conv.bias is not None:
            new_conv.bias.data = select_channels(old_conv.bias.data, remove_channels, dim)
        return new_conv
    else:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels - len(remove_channels), out_channels=old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data, residue = select_channels(old_conv.weight.data, remove_channels, dim)

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
        return new_conv, residue

def get_mobile_conv(old_conv, remove_channels, dim):
    new_conv = nn.Conv2d(in_channels=old_conv.out_channels - len(remove_channels),
                         out_channels=old_conv.out_channels - len(remove_channels),
                         kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                         dilation=old_conv.dilation, bias=old_conv.bias is not None)
    new_conv.weight.data = select_channels(old_conv.weight.data, remove_channels, dim)

    if old_conv.bias is not None:
        new_conv.bias.data = select_channels(old_conv.bias.data, remove_channels, dim)
    return new_conv

def get_new_norm(old_norm, remove_channels):
    new = torch.nn.BatchNorm2d(num_features=old_norm.num_features - len(remove_channels), eps=old_norm.eps,
                               momentum=old_norm.momentum, affine=old_norm.affine,
                               track_running_stats=old_norm.track_running_stats)
    new.weight.data = select_channels(old_norm.weight.data, remove_channels, 0)
    new.bias.data = select_channels(old_norm.bias.data, remove_channels, 0)

    if old_norm.track_running_stats:
        new.running_mean.data = select_channels(old_norm.running_mean.data, remove_channels, 0)
        new.running_var.data = select_channels(old_norm.running_var.data, remove_channels, 0)

    return new


def get_new_linear(old_linear, remove_channels):
    new = torch.nn.Linear(in_features=old_linear.in_features - len(remove_channels),
                          out_features=old_linear.out_features, bias=old_linear.bias is not None)
    new.weight.data, residue = select_channels(old_linear.weight.data, remove_channels, 1)
    if old_linear.bias is not None:
        new.bias.data = old_linear.bias.data
    return new
