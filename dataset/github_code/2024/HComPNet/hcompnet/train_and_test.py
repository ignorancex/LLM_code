from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math
import numpy as np

from collections import defaultdict
from torchmetrics.functional import f1_score, recall, precision
from torchvision.datasets.folder import ImageFolder

import os
import gc

def train(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, \
          epoch, device, pretrain=False, finetune=False, wandb_logging=True, \
          wandb_run=None, pretrain_epochs=0, args=None):
    run_epoch(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, \
                 epoch, device, pretrain=pretrain, finetune=finetune, progress_prefix = 'Train Epoch', wandb_logging=True, \
                 wandb_run=None, pretrain_epochs=pretrain_epochs, args=args, train=True)


def test(net, test_loader, criterion, epoch, device, pretrain=False, finetune=False, wandb_logging=True, \
          wandb_run=None, pretrain_epochs=0, args=None):
    run_epoch(net, test_loader, optimizer_net=None, optimizer_classifier=None, scheduler_net=None, scheduler_classifier=None, criterion=criterion, \
                 epoch=epoch, device=device, pretrain=pretrain, finetune=finetune, progress_prefix = 'Test Epoch', wandb_logging=True, \
                 wandb_run=None, pretrain_epochs=pretrain_epochs, args=args, train=False)
    

def run_epoch(net, data_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, \
                 epoch, device, pretrain=False, finetune=False, progress_prefix: str = 'None', wandb_logging=True, \
                 wandb_run=None, pretrain_epochs=0, args=None, train=True):
    epoch_info = {}

    root = net.module.root
    dataset = data_loader.dataset
    while type(dataset) != ImageFolder:
        dataset = dataset.dataset
    name2label = dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    
    # For debugging purpose
    node_accuracy = {}
    for node in root.nodes_with_children():
        node_accuracy[node.name] = {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'f1': None, 'preds': torch.empty(0, node.num_children()).cpu(), 'gts': torch.empty(0).cpu()}
        node_accuracy[node.name]['children'] = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})

    net.train()

    if pretrain:
        # Disable training of classification layer
        for attr in dir(net.module):
            if attr.endswith('_classification'):
                getattr(net.module, attr).requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        for attr in dir(net.module):
            if attr.endswith('_classification'):
                getattr(net.module, attr).requires_grad = True

    iters = len(data_loader)
    # Show progress on progress bar. 
    data_iter = tqdm(enumerate(data_loader),
                    total=len(data_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)

    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/pretrain_epochs)*1.
        t_weight = 5.
    else:
        align_pf_weight = 5. 
        t_weight = 2.

    cl_weight = float(args.cl_weight)
    ovsp_weight = float(args.ovsp_weight)
    orth_weight = float(args.orth_weight)
    disc_weight = float(args.disc_weight)

    n_fine_correct = 0
    n_samples = 0

    # Iterate through the data set to update leaves, prototypes and network
    for i, data in data_iter:       
        
        # xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
        if train:
            xs1, xs2, ys = data
            xs = torch.cat([xs1, xs2]).to(device)
            ys = torch.cat([ys, ys]).to(device)
        else:
            xs, ys = data
            xs = torch.cat([xs, xs]).to(device)
            ys = torch.cat([ys, ys]).to(device)

        grad_req = torch.enable_grad() if train else torch.no_grad()

        with grad_req:

            if train:
                # Reset the gradients
                optimizer_classifier.zero_grad(set_to_none=True)
                optimizer_net.zero_grad(set_to_none=True)

            features, proto_features, pooled, out = net(xs)

            loss = calculate_loss(epoch, net, features, proto_features, pooled, out, ys, net.module._multiplier, \
                                    align_pf_weight, t_weight, cl_weight, ovsp_weight, orth_weight, disc_weight, \
                                    pretrain, finetune, criterion, data_iter=data_iter, EPS=1e-10, root=root, \
                                    label2name=label2name, node_accuracy=node_accuracy, train=True, args=args, device=device)

            # Compute the gradient
            if train:
                loss.backward()

            if (not pretrain) and (train):
                optimizer_classifier.step()   
                scheduler_classifier.step(epoch - 1 + (i/iters))
        
            if (not finetune) and (train):
                optimizer_net.step()
                scheduler_net.step() 

        if not pretrain:
            with torch.no_grad():
                for attr in dir(net.module):
                    if attr.endswith('_classification'):
                        classification_layer = getattr(net.module, attr)
                        classification_layer.weight.copy_(torch.clamp(classification_layer.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                        if classification_layer.bias is not None:
                            classification_layer.bias.copy_(torch.clamp(classification_layer.bias.data, min=0.))  

        _, preds_joint = net.module.get_joint_distribution(out)
        _, fine_predicted = torch.max(preds_joint.data, 1)
        fine_correct = fine_predicted == ys
        n_fine_correct += fine_correct.sum().item()
        n_samples += ys.size(0)
    
    fine_accuracy = n_fine_correct/n_samples
    epoch_info['fine_accuracy'] = fine_accuracy
    
    # Uncomment to print node wise accuracy. Commented since there are many nodes
    # for node_name in node_accuracy:
    #     node_accuracy[node_name]['accuracy'] = round((node_accuracy[node_name]['n_correct'] / node_accuracy[node_name]['n_examples']) * 100, 2)
    #     node_accuracy[node_name]['f1'] = f1_score(node_accuracy[node_name]["preds"], node_accuracy[node_name]["gts"].to(torch.int), average='weighted', num_classes=net.module.root.get_node(node_name).num_children()).item()
    #     node_accuracy[node_name]['f1'] = round(node_accuracy[node_name]['f1'] * 100, 2)
    print(epoch_info)
    return epoch_info


def calculate_loss(epoch, net, features, proto_features, pooled, out, ys, net_normalization_multiplier, \
                    align_pf_weight, t_weight, cl_weight, ovsp_weight, orth_weight, disc_weight, \
                    pretrain, finetune, criterion, data_iter, EPS=1e-10, root=None, \
                    label2name=None, node_accuracy=None, train=True, args=None, device=None):
    mask_loss = {}
    disc_loss = {}
    a_loss_pf = {}
    tanh_loss = {}
    ovsp_loss = {}
    orth_loss = {}
    class_loss = {}
    losses_used = set()
    loss = 0

    batch_names = [label2name[y.item()] for y in ys]
    features1, features2 = features.chunk(2)

    for node in root.nodes_with_children():
        children_idx = torch.tensor([name in node.leaf_descendents for name in batch_names])
        batch_names_coarsest = [node.closest_descendent_for(name).name for name in batch_names if name in node.leaf_descendents]
        node_y = torch.tensor([node.children_to_labels[name] for name in batch_names_coarsest]).to(device)#.cuda()

        if len(node_y) == 0:
            continue

        node_logits = out[node.name][children_idx]
        
        if (not finetune):
            pf1, pf2 = proto_features[node.name][children_idx].chunk(2)
            embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
            embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
            a_loss_pf[node.name] = (align_loss(embv1, embv2.detach()) \
                                    + align_loss(embv2, embv1.detach())) / 2.
            loss += align_pf_weight * a_loss_pf[node.name] / len(root.nodes_with_children())
            losses_used.add('LA')

            pooled1, pooled2 = pooled[node.name][children_idx].chunk(2)
            tanh_loss[node.name] = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() \
                                    + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean()) / 2.
            loss += t_weight * tanh_loss[node.name] / len(root.nodes_with_children())
            losses_used.add('LT')
        
        # Masking module
        if (not pretrain):
            proto_presence = getattr(net.module, '_'+node.name+'_proto_presence')
            overspecificity_score_current_node = 0.
            mask_l1_loss_current_node = 0.
            total_num_relevant_protos = 0. # sum of relevant protos for each class, no proto is relevant to more than one class so this would work
            
            for child_node in node.children:
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                child_class_idx = node.children_to_labels[child_node.name]
                relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                num_relevant_protos = relevant_proto_idx.shape[0]
                total_num_relevant_protos += num_relevant_protos

                max_pooled_each_descendant = torch.empty(0, num_relevant_protos).to(device)#.cuda()
                for descendant_name in child_node.leaf_descendents:
                    descendant_idx = torch.tensor([name == descendant_name for name in batch_names])
                    if torch.sum(descendant_idx).item() == 0: # no of descendants in batch is zero
                        continue
                    max_vals, _ = torch.max(pooled[node.name][descendant_idx][:, relevant_proto_idx], dim=0, keepdim=True) # [1, num_relevant_protos]
                    max_pooled_each_descendant = torch.cat([max_pooled_each_descendant, max_vals], dim=0)

                if max_pooled_each_descendant.shape[0] == 0: # if none of the descendants are in the batch
                    continue

                proto_presence = F.gumbel_softmax(proto_presence, tau=0.5, hard=False, dim=-1)

                overspecificity_score_current_node += (-1) * (torch.prod(torch.clamp(max_pooled_each_descendant.clone().detach() * 1.1, max=1.0), dim=0) * proto_presence[relevant_proto_idx, 1]).sum()
                mask_l1_loss_current_node += proto_presence[relevant_proto_idx, 1].sum()

            overspecificity_score_current_node /= total_num_relevant_protos
            mask_l1_loss_current_node /= total_num_relevant_protos

            mask_loss[node.name] = ((2.0 * overspecificity_score_current_node) + (0.5 * mask_l1_loss_current_node)) / len(root.nodes_with_children())

            loss += mask_loss[node.name]
            losses_used.add('L_MASK')

        # Discrimination loss
        if (not pretrain) and (not finetune):
            num_protos = pooled[node.name].shape[-1]
            classification_weights = getattr(net.module, '_'+node.name+'_classification').weight

            node_y_expanded = node_y.unsqueeze(-1).repeat(1, num_protos) # [batch_size] to [batch_size, num_protos]
            disc_loss[node.name] = 0.
            max_of_contrasting_set = torch.empty(1, 0).to(device)
            for child_node in node.children:
                child_class_idx = node.children_to_labels[child_node.name]
                relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-5).squeeze(-1)

                if len(relevant_proto_idx) == 0:
                    # likely to happen when leave_out_classes is used
                    assert child_node.is_leaf()
                    assert args.leave_out_classes.strip() != '' 
                    continue

                # look at data points that not descendant of this child node
                relevant_data_idx = torch.nonzero(node_y != child_class_idx).squeeze(-1)
                if len(relevant_data_idx) == 0:
                    continue # no data points that is not descendant of this child node present in this batch so skip this child_node
                
                max_of_contrast, topk_idx = torch.topk(pooled[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx], dim=0, k=1)
                max_of_contrasting_set = torch.cat([max_of_contrasting_set, max_of_contrast], dim=1)

            if max_of_contrasting_set.numel() != 0:
                disc_loss[node.name] = disc_weight * max_of_contrasting_set.mean() / len(root.nodes_with_children())
                loss += disc_loss[node.name]
                losses_used.add('DISC_L')

        # Prototypes orthogonality loss
        if (not pretrain) and (not finetune):
            prototype_kernels = getattr(net.module, '_'+node.name+'_add_on')
            classification_layer = getattr(net.module, '_'+node.name+'_classification')
            relevant_prototype_kernels = prototype_kernels.weight[(classification_layer.weight > 0.001).any(dim=0)]
            orth_loss[node.name] = orth_dist(relevant_prototype_kernels, device=device)
            loss += orth_weight * orth_loss[node.name] / len(root.nodes_with_children())
            losses_used.add('L_ORTH')

        # Cross entropy loss
        if (not pretrain):
            softmax_inputs = torch.log1p(node_logits**net_normalization_multiplier)
            class_loss[node.name] = criterion(softmax_inputs, node_y, node.weights) 
            loss += cl_weight * class_loss[node.name] / len(root.nodes_with_children())
            losses_used.add('LC')

        # Overspecificity loss
        if (not finetune) and (not pretrain):
            ovsp_for_each_descendant = []
            for child_node in node.children:
                child_class_idx = node.children_to_labels[child_node.name]
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                for descendant_name in child_node.leaf_descendents:
                    descendant_idx = torch.tensor([name == descendant_name for name in batch_names])
                    relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                    if len(relevant_proto_idx) == 0:
                        assert args.leave_out_classes.strip() != '' # this is likely to happen when using leave_out_classes
                        assert child_node.is_leaf()
                        continue
                    descendant_pooled1, descendant_pooled2 = pooled[node.name][descendant_idx][:, relevant_proto_idx].chunk(2)
                    descendant_ovsp_loss = -(torch.log(torch.tanh(torch.sum(descendant_pooled1,dim=0))+EPS).mean() \
                                                            + torch.log(torch.tanh(torch.sum(descendant_pooled2,dim=0))+EPS).mean()) / 2.
                    ovsp_for_each_descendant.append(descendant_ovsp_loss)
            ovsp_loss[node.name] = torch.mean(torch.stack(ovsp_for_each_descendant), dim=0)
            loss += ovsp_weight * ovsp_loss[node.name] / len(root.nodes_with_children())
            losses_used.add('L_OVSP')

        # For debugging purpose. Nodewise accuracy
        node_accuracy[node.name]['n_examples'] += node_y.shape[0]
        _, node_coarsest_predicted = torch.max(node_logits.data, 1)
        node_accuracy[node.name]['n_correct'] += (node_y == node_coarsest_predicted).sum().item()
        for child in node.children:
            node_accuracy[node.name]['children'][child.name]['n_examples'] += (node_y == node.children_to_labels[child.name]).sum().item()
            node_accuracy[node.name]['children'][child.name]['n_correct'] += (node_coarsest_predicted[node_y == node.children_to_labels[child.name]] == node.children_to_labels[child.name]).sum().item()
            node_accuracy[node.name]['preds'] = torch.cat((node_accuracy[node.name]['preds'], node_logits.detach().cpu())).detach().cpu()#.numpy()
            node_accuracy[node.name]['gts'] = torch.cat((node_accuracy[node.name]['gts'], node_y.detach().cpu())).detach().cpu()#.numpy()

    avg_mask_loss = torch.tensor(list(mask_loss.values())).mean().item() if mask_loss else -5
    avg_disc_loss = torch.tensor(list(disc_loss.values())).mean().item() if disc_loss else -5
    avg_a_loss_pf = torch.tensor(list(a_loss_pf.values())).mean().item() if a_loss_pf else -5
    avg_tanh_loss = torch.tensor(list(tanh_loss.values())).mean().item() if tanh_loss else -5
    avg_ovsp_loss = torch.tensor(list(ovsp_loss.values())).mean().item() if ovsp_loss else -5
    avg_orth_loss = torch.tensor(list(orth_loss.values())).mean().item() if orth_loss else -5
    avg_class_loss = torch.tensor(list(class_loss.values())).mean().item() if class_loss else -5
    
    if pretrain:
        data_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA_PF:{avg_a_loss_pf:.2f}, LT:{avg_tanh_loss:.3f}, losses_used:{"+".join(losses_used)}', refresh=False)
    else:
        data_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LC: {avg_class_loss:.3f}, LA_PF:{avg_a_loss_pf:.2f}, LT:{avg_tanh_loss:.3f}, L_OVSP:{avg_tanh_loss:.3f}, L_DISC:{avg_disc_loss:.3f}, L_MASK:{avg_mask_loss:.3f}, L_ORTH:{avg_orth_loss:.3f}, losses_used:{"+".join(losses_used)}', refresh=False)
    
    return loss


def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss


def orth_dist(mat, stride=None, device=None):
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).to(device))