import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import load_data, validate, model_parameter_vector
import copy
from nodes import Node
import random
from sam import SAM, SAMFED
from utils import trainable_params, cuda

##############################################################################
# General client function 
##############################################################################

def receive_server_model(args, client_nodes, central_node):

    for idx in range(len(client_nodes)):
        if 'fedawo' in args.server_method:
            client_nodes[idx].model.load_param(copy.deepcopy(central_node.model.get_param(clone = True)))
        elif args.client_method == 'mask_intrinsic':
            client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
            client_nodes[idx].model.initial_value = copy.deepcopy(central_node.model.initial_value)
            client_nodes[idx].model.random_matrix = copy.deepcopy(central_node.model.random_matrix)
            client_nodes[idx].model.m[0].load_state_dict(copy.deepcopy(central_node.model.m[0].state_dict()))

        elif args.server_method == 'none':
            pass

        else:
            client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
        
    return client_nodes

def Client_update(args, client_nodes, central_node, select_list):
    '''
    client update functions
    '''
    # clients receive the server model 
    client_nodes = receive_server_model(args, client_nodes, central_node)

    # update the global model
    if args.client_method in ['local_train', 'mask_projection', 'pruning', 'mask_activation', 'mask_fastslow', 'mask_direction', 'mask_gaussian', 'mask_intrinsic']:
        if args.client_method in ['mask_activation', 'pruning', 'mask_fastslow', 'mask_fixed']:
            mask = central_node.mask
        elif args.client_method == 'mask_direction':
            mask = central_node.direc_mask
        elif args.client_method == 'mask_gaussian':
            mask = central_node.gauss_mask

        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                if args.client_method in ['local_train', 'mask_projection', 'mask_intrinsic']:
                    loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
    
    elif args.client_method == 'fedsam':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedsam(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
    
    elif args.client_method == 'fedlc':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedlc(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'fedprox':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedprox(global_model_param, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
    
    elif args.client_method in ('multi_step_group_connectivity', 'multi_step_group_connectivity_plus'):
        total_model_num = len(central_node.global_model_list)
        if total_model_num >= args.group_model_num:
            group_model_num = args.group_model_num
        else:
            group_model_num = total_model_num

        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = multi_step_client_group_connectivity(group_model_num, central_node.global_model_list, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'feddyn':
        global_model_vector = copy.deepcopy(model_parameter_vector(args, central_node.model).detach().clone())
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_feddyn(global_model_vector, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

            # update old grad
            v1 = model_parameter_vector(args, client_nodes[i].model).detach()
            client_nodes[i].old_grad = client_nodes[i].old_grad - args.mu * (v1 - global_model_vector)

    elif args.client_method == 'fedrod':
        # print("fedrod")
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedrod(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
    
    elif args.client_method == 'moon':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_moon(args, client_nodes[i], central_node.global_model_list)
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'scaffold':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_scaffold(args, client_nodes[i], central_node)
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    else:
        raise ValueError('Undefined client method...')

    return client_nodes, train_loss


def freeze_layers(model, layers_to_freeze):
    for name, p in model.named_parameters():
        try:
            if name in layers_to_freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True
        except:
            pass
    return model

def unfreeze_all_layers(model):
    for name, p in model.named_parameters():
        try:
            p.requires_grad = True
        except:
            pass
    return model


def Client_validate(args, client_nodes, select_list):
    '''
    client validation functions, for testing local personalization
    '''
    client_acc = []
    for idx in select_list:
        acc = validate(args, client_nodes[idx])
        print('client ', idx, ', after  training, acc is', acc)
        client_acc.append(acc)
    avg_client_acc = sum(client_acc) / len(client_acc)

    return avg_client_acc

def client_localTrain(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    if args.sam:
        sam_optimizer = SAMFED(node.optimizer, node.model, args.sam_rho, 0)
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        if args.sam:
            node.optimizer.zero_grad()
        else:
            node.optimizer.zero_grad()
        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)

        loss_local.backward()
        loss = loss + loss_local.item()
        
        if args.sam: 
            sam_optimizer.ascent_step()
            
            loss_second = F.cross_entropy(node.model(data), target)
            loss_second.backward()
            sam_optimizer.descent_step()
        else:
            node.optimizer.step()

    return loss/len(train_loader)

def client_fedlc(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local = balanced_softmax_loss_zj(output_local, target, node.class2data)

        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

def client_fedsam(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    sam_optimizer = SAMFED(node.optimizer, node.model, args.sam_rho, 0)
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)

        loss_local.backward()
        loss = loss + loss_local.item()
        
        sam_optimizer.ascent_step()
        
        loss_second = F.cross_entropy(node.model(data), target)
        loss_second.backward()
        sam_optimizer.descent_step()

    return loss/len(train_loader)

def multi_step_client_group_connectivity(group_model_num, global_model_list, args, node, loss = 0.0):
    node.model.train()

    interpolated_model = copy.deepcopy(global_model_list[-1]).requires_grad_(True)

    loss = 0.0
    train_loader = node.local_data  # iid
    if args.sam:
        sam_optimizer = SAMFED(node.optimizer, node.model, args.sam_rho, 0)
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()

        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local = node.model(data)

        loss_local =  loss_F(args, node, output_local, target)
        loss_local.backward()
        if args.sam: 
            sam_optimizer.ascent_step()
            
            loss_second = loss_F(args, node, node.model(data), target)
            loss_second.backward()
            sam_optimizer.descent_step()
        else:
            node.optimizer.step()

        # loss for improving connectivity
        alpha = 0.5
        beta = args.beta
        for idx in range(1, group_model_num + 1):
            node.optimizer.zero_grad()
            interpolated_model.zero_grad()

            global_model = copy.deepcopy(global_model_list[-idx])
            global_model.requires_grad_(False)

            for param, param_fixed, param_interp in zip(node.model.parameters(), global_model.parameters(), interpolated_model.parameters()):
                param_interp.data = alpha * param.data + (1 - alpha) * param_fixed.data
            
            output_interp = interpolated_model(data)
            loss_interp = loss_F(args, node, output_interp, target)
            loss_interp.backward()

            # adding the interpolated grad to the model
            for param, param_interp in zip(node.model.parameters(), interpolated_model.parameters()):
                if param.grad is None and param_interp.grad is None:
                    continue
                if param.grad is not None:
                    param.grad = param.grad + beta*alpha/group_model_num*param_interp.grad
                else:
                    param.grad = beta*alpha/group_model_num*param_interp.grad

            if args.sam: 
                sam_optimizer.ascent_step()
                
                for param, param_fixed, param_interp in zip(node.model.parameters(), global_model.parameters(), interpolated_model.parameters()):
                    param_interp.data = alpha * param.data + (1 - alpha) * param_fixed.data
                
                output_interp = interpolated_model(data)
                loss_interp = loss_F(args, node, output_interp, target)
                loss_interp.backward()

                # adding the interpolated grad to the model
                for param, param_interp in zip(node.model.parameters(), interpolated_model.parameters()):
                    if param.grad is None and param_interp.grad is None:
                        continue
                    if param.grad is not None:
                        param.grad = param.grad + beta*alpha/group_model_num*param_interp.grad
                    else:
                        param.grad = beta*alpha/group_model_num*param_interp.grad
                
                sam_optimizer.descent_step()
            else:
                node.optimizer.step()
            
        loss = loss + loss_local.item()

        

    return loss/len(train_loader)

def client_fedprox(global_model_param, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        # fedprox update
        node.optimizer.step(global_model_param)

    return loss/len(train_loader)

def client_feddyn(global_model_vector, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss = loss + loss_local.item()

        # feddyn update
        v1 = model_parameter_vector(args, node.model)
        loss_local += args.mu/2 * torch.norm(v1 - global_model_vector, 2)
        loss_local -= torch.dot(v1, node.old_grad)

        loss_local.backward()
        node.optimizer.step()

    return loss/len(train_loader)

def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def balanced_softmax_loss_zj(logits, lables, class2data, calibration_temp=0.1):
    logits -= calibration_temp * class2data**(-0.25)
    loss = F.cross_entropy(input=logits, target=lables)
    return loss

def loss_F(args, node: Node, output, target):
    if args.is_balanced_loss:
        return balanced_softmax_loss_zj(output, target, node.class2data)
    else:
        return F.cross_entropy(output, target)

def client_fedrod(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data

    # initialize the optimizer of p_head
    p_head_optimizer = torch.optim.SGD(node.p_head.parameters(), lr=args.lr)

    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        p_head_optimizer.zero_grad()

        # train model
        
        data, target = cuda(data), target.cuda()
        _, logit_g, feature = node.model(data, return_hidden_state=True)

        # balanced loss for base and g_head
        loss_local = balanced_softmax_loss(logit_g, target, node.sample_per_class)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

        # ce loss for p_head
        logit_p = node.p_head(feature.detach())
        logit = logit_g.detach() + logit_p
        loss_p =  F.cross_entropy(logit, target)
        loss_p.backward()
        p_head_optimizer.step()

    return loss/len(train_loader)

def client_moon(args, node: Node, global_model_list, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    global_model = copy.deepcopy(global_model_list[-1])
    cos_sim = torch.nn.CosineSimilarity(dim=-1)
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local, features = node.model(data, return_features=True)
        _, features_global = global_model(data, return_features=True)
        _, features_prev_local = node.prev_model(data, return_features=True)

        loss_local =  F.cross_entropy(output_local, target)

        # contrastive loss
        # similarity of positive pair (i.e., w/ global model)
        pos_similarity = cos_sim(features, features_global).view(-1, 1)
        # similarity of negative pair (i.e., w/ previous round local model)
        neg_similarity = cos_sim(features, features_prev_local).view(-1, 1)
        repres_sim = torch.cat([pos_similarity, neg_similarity], dim=-1)
        contrast_label = torch.zeros(repres_sim.size(0)).long().cuda()
        loss_con = F.cross_entropy(repres_sim, contrast_label)
        loss_cur = loss_local + loss_con

        loss_cur.backward()
        loss = loss + loss_cur.item()
        
        node.optimizer.step()

    return loss/len(train_loader)

def client_scaffold(args, node: Node, central_node: Node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    global_model: nn.Module = copy.deepcopy(central_node.global_model_list[-1])
    node.c_global = central_node.c_global
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = load_data(args, node.cluster_id, data, target)
        data, target = cuda(data), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)

        loss_local.backward()
        loss = loss + loss_local.item()

        for param, c, c_i in zip(
                trainable_params(node.model),
                node.c_global,
                node.c_local,
            ):
                param.grad.data += c - c_i
        
        node.optimizer.step()
    
    # update local variate
    with torch.no_grad():
        y_delta = []
        c_plus = []
        c_delta = []

        # compute y_delta (difference of model before and after training)
        for x, y_i in zip(trainable_params(global_model), trainable_params(node.model)):
            y_delta.append(y_i - x)

        # compute c_plus
        coef = 1 / (args.E * args.lr)
        for c, c_i, y_del in zip(
            node.c_global, node.c_local, y_delta
        ):
            c_plus.append(c_i - c - coef * y_del)

        # compute c_delta
        for c_p, c_l in zip(c_plus, node.c_local):
            c_delta.append(c_p - c_l)

        node.c_local = c_plus
        node.y_delta = y_delta
        node.c_delta = c_delta

    return loss/len(train_loader)