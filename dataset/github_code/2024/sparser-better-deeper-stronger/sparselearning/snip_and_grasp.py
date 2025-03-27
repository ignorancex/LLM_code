import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import types



def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device, loss="logsoftmax"):

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs.requires_grad = True
    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net).to(device)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            #nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    if loss == "logsoftmax":
        loss = F.nll_loss(outputs, targets)
    elif loss == "logits":
        loss = F.cross_entropy(outputs, targets)
    else:
        raise ValueError("Unknonw loss")
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    layer_wise_sparsities = []
    layer_wise_mask = []
    for g in grads_abs:
        mask = ((g / norm_factor) >= acceptable_score).float()
        sparsity = float((mask==0).sum().item() / mask.numel())
        layer_wise_sparsities.append(sparsity)
        layer_wise_mask.append(mask.detach())
    print(f'layer-wise sparsity is {layer_wise_sparsities}')

    return layer_wise_sparsities, layer_wise_mask

def SNIP_training(net, keep_ratio, train_dataloader, device, masks, death_rate):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    print('Pruning rate:', death_rate)
    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)
    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    # for layer in net.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
    #         # nn.init.xavier_normal_(layer.weight)
    #         # layer.weight.requires_grad = False
    #
    #     # Override the forward methods:
    #     if isinstance(layer, nn.Conv2d):
    #         layer.forward = types.MethodType(snip_forward_conv2d, layer)
    #
    #     if isinstance(layer, nn.Linear):
    #         layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    grads_abs = []
    masks_copy = []
    new_masks = []
    for name in masks:
        masks_copy.append(masks[name])

    index = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # clone mask
            mask = masks_copy[index].clone()

            num_nonzero = (masks_copy[index] != 0).sum().item()
            num_zero = (masks_copy[index] == 0).sum().item()

            # calculate score
            scores = torch.abs(layer.weight.grad * layer.weight * masks_copy[index]) # weight * grad
            norm_factor = torch.sum(scores)
            scores.div_(norm_factor)

            x, idx = torch.sort(scores.data.view(-1))
            num_remove = math.ceil(death_rate * num_nonzero)
            k = math.ceil(num_zero + num_remove)
            if num_remove == 0.0: return masks_copy[index] != 0.0

            mask.data.view(-1)[idx[:k]] = 0.0

            new_masks.append(mask)
            index += 1

    return new_masks


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


def GraSP(model, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = ratio
    old_net = model

    net = copy.deepcopy(model)  # .eval()
    net.zero_grad()

    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    print_once = False
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N//2])
        targets_one.append(dtarget[:N//2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net.forward(inputs[:N//2])/T
        if print_once:
            # import pdb; pdb.set_trace()
            x = F.softmax(outputs)
            print(x)
            print(x.max(), x.min())
            print_once = False
        loss = F.cross_entropy(outputs, targets[:N//2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = net.forward(inputs[N // 2:])/T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs)/T
        loss = F.cross_entropy(outputs, targets)
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)

    layer_wise_sparsities = []
    layer_wise_mask = []
    for m, g in grads.items():
        mask = ((g / norm_factor) < acceptable_score).float()
        sparsity = float((mask == 0).sum().item() / mask.numel())
        layer_wise_sparsities.append(sparsity)
        layer_wise_mask.append(mask.detach())

    print(f'layer-wise sparsity is {layer_wise_sparsities}')

    return layer_wise_sparsities, layer_wise_mask


def GraSP_v2(model, keep_ratio, dataloader, device, masked_parameters, temp = 200, eps = 1e-10):

    # first gradient vector without computational graph
    stopped_grads = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data) / temp
        L = F.cross_entropy(output, target)

        grads = torch.autograd.grad(L, [p for (_, p) in masked_parameters], create_graph=False)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        stopped_grads += flatten_grads

    # second gradient vector with computational graph
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data) / temp
        L = F.cross_entropy(output, target)

        grads = torch.autograd.grad(L, [p for (_, p) in masked_parameters], create_graph=True)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

        gnorm = (stopped_grads * flatten_grads).sum()
        gnorm.backward()

    scores = {}
    # calculate score Hg * theta
    for name, p in masked_parameters:
        scores[name] = torch.clone(p.grad * p.data).detach()
        p.grad.data.zero_()

    # normalize score
    all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
    norm = torch.abs(torch.sum(all_scores)) + eps
    all_scores.div_(norm)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()

    if len(threshold)>0:
        acceptable_score = threshold[-1]
    else:
        acceptable_score = torch.max(all_scores)

    print('** accept: ', acceptable_score)

    layer_wise_sparsities = []
    layer_wise_mask = []
    for name, p in masked_parameters:
        mask = (scores[name]/norm < acceptable_score).float()
        sparsity = float((mask == 0).sum().item() / mask.numel())
        layer_wise_sparsities.append(sparsity)
        layer_wise_mask.append(mask.detach())

    print(f'layer-wise sparsity is {layer_wise_sparsities}')

    return layer_wise_sparsities, layer_wise_mask



