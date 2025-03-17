import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from utils.utils_dh import AverageMeter
from utils.test_utils import NN, kNN_DA, recompute_memory

import logging

from collections import OrderedDict


def train_selfsup_only(
    epoch, args, wandb, net, lemniscate_s, 
    lemniscate_t, optimizer, 
    source_loader, target_loader_unl, 
    source_syn_loader=None, target_syn_loader=None):
    
    cross_entropy_loss = nn.CrossEntropyLoss()

    net.eval()
    if lemniscate_s.memory_first:
        recompute_memory(epoch, net, lemniscate_s, source_loader)

    if lemniscate_t.memory_first:
        recompute_memory(epoch, net, lemniscate_t, target_loader_unl)
        
    net.train()

    start = time.time()

    scaler = torch.cuda.amp.GradScaler()
    
    avg_meters = OrderedDict({
        'Loss': AverageMeter(),
        'Loss CDM': AverageMeter(),
        'Source Cross Entropy': AverageMeter(),
        'Target Cross Entropy': AverageMeter(),
    })

    if args.ppp:
        assert source_syn_loader is not None and target_syn_loader is not None, \
            'source_syn_loader and target_syn_loader should not be None'
        avg_meters.update({
            'Source PPP Loss': AverageMeter(),
        })
        avg_meters.update({
            'Target PPP Loss': AverageMeter(),
        })

    for batch_idx, (inputs, _, indexes) in enumerate(source_loader):

        try:
            inputs2, _, indexes2 = next(target_loader_unl_iter)
        except:
            target_loader_unl_iter = iter(target_loader_unl)
            inputs2, _, indexes2 = next(target_loader_unl_iter)
            
        if args.ppp:
            try:
                inputs_syn, _, indexes_syn = next(source_syn_loader_iter)
            except:
                source_syn_loader_iter = iter(source_syn_loader)
                inputs_syn, _, indexes_syn = next(source_syn_loader_iter)
            assert (indexes == indexes_syn).all(), "indexes and indexes_syn should be same"
            inputs_syn = inputs_syn.cuda(non_blocking=True)
            indexes_syn = indexes_syn.to(dtype=torch.long, device='cuda', non_blocking=True)
            
            try:
                inputs2_syn, _, indexes2_syn = next(target_syn_loader_iter)
            except:
                target_syn_loader_iter = iter(target_syn_loader)
                inputs2_syn, _, indexes2_syn = next(target_syn_loader_iter)
                
            assert (indexes2 == indexes2_syn).all(), "indexes2 and indexes2_syn should be same"
            inputs2_syn = inputs2_syn.cuda(non_blocking=True)
            indexes2_syn = indexes2_syn.to(dtype=torch.long, device='cuda', non_blocking=True)

        inputs = inputs.cuda(non_blocking=True)
        indexes = indexes.to(dtype=torch.long, device='cuda', non_blocking=True)
        inputs2 = inputs2.cuda(non_blocking=True)
        indexes2 = indexes2.to(dtype=torch.long, device='cuda', non_blocking=True)
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            features1 = net(inputs)
            features2 = net(inputs2)

            outputs = lemniscate_s(features1)
            outputs2 = lemniscate_t(features2)

            loss_id = 0

            source_cross = cross_entropy_loss(outputs, indexes)
            target_cross = cross_entropy_loss(outputs2, indexes2)

            loss_id = (source_cross + target_cross) / 2.0

            total_loss = loss_id

            if not args.no_cross:
                outputs4 = lemniscate_s(features2)
                outputs4 = torch.topk(outputs4, min(args.n_neighbor, len(source_loader.dataset)), dim=1)[0]
                outputs4 = F.softmax(outputs4, dim=1)
                loss_ent4 = -1. * torch.mean(torch.sum(outputs4 * (torch.log(outputs4 + 1e-5)), 1))
                loss_cdm = loss_ent4

                
                outputs3 = lemniscate_t(features1)
                outputs3 = torch.topk(outputs3,  min(args.n_neighbor, len(target_loader_unl.dataset)), dim=1)[0]
                outputs3 = F.softmax(outputs3, dim=1)
                loss_ent3 = -1. * torch.mean(torch.sum(outputs3 * (torch.log(outputs3 + 1e-5)), 1))
                loss_cdm += loss_ent3
                total_loss += loss_cdm

            if args.ppp:
                features1_syn = net(inputs_syn)
                outputs5 = torch.mm(features1_syn, features1.t())/args.T_syn
                loss_syn_s = F.cross_entropy(outputs5, torch.arange(len(features1), device=outputs5.device, dtype=torch.long))
                total_loss += args.syn_wt * loss_syn_s
            
                features2_syn = net(inputs2_syn)
                outputs6 = torch.mm(features2_syn, features2.t())/args.T_syn
                loss_syn_t = F.cross_entropy(outputs6, torch.arange(len(features2), device=outputs6.device, dtype=torch.long))
                total_loss += args.syn_wt * loss_syn_t
                
            avg_meters['Loss'].update(total_loss.item(), inputs.size(0))
            if not args.no_cross:
                avg_meters['Loss CDM'].update(loss_cdm.item(), inputs.size(0))
            avg_meters['Source Cross Entropy'].update(source_cross.item(), inputs.size(0))
            avg_meters['Target Cross Entropy'].update(target_cross.item(), inputs.size(0))
            
            if args.ppp:
                avg_meters['Source PPP Loss'].update(loss_syn_s.item(), inputs.size(0))
                avg_meters['Target PPP Loss'].update(loss_syn_t.item(), inputs.size(0))
            
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)

        scaler.update()
        optimizer.zero_grad()

        lemniscate_s.update_weight(features1.detach(), indexes)
        lemniscate_t.update_weight(features2.detach(), indexes2)

        if (batch_idx+1) % args.log_freq == 0:
            logging.info('Epoch: [{}][{}/{}] Loss:{:.4f}'.format(epoch, batch_idx, len(source_loader), loss_id.item()))
            logging.info('Cross Entropy : source:{:.4f}  target:{:.4f}'.format(source_cross.item(), target_cross.item()))
                        
            log_info = OrderedDict({
                'Loss': loss_id.item(),
                'Source Cross Entropy': source_cross.item(),
                'Target Cross Entropy': target_cross.item(),
            })
            if not args.no_cross:
                log_info.update({
                    'Loss CDM': loss_cdm.item(),
                })
            if args.ppp:
                log_info.update({
                    'Source PPP Loss': loss_syn_s.item(),
                    'Target PPP Loss': loss_syn_t.item(),
                })
            wandb.log(log_info)

    end = time.time()
    logging.info('Epoch time: {}'.format(end-start))

    lemniscate_s.memory_first = False
    lemniscate_t.memory_first = False

    return OrderedDict({f'Epoch Avg {k}' : avg_meters[k].avg for k in avg_meters})

