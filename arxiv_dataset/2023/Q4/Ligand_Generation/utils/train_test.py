import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, loader, optimizer, loss_fn, device, evaluator = None):
    model.train()
    meter = AverageMeter()
    for batch in tqdm(loader, total = len(loader)): 
        optimizer.zero_grad()
        protein_graph, seq_emb, ligand_graph, label = batch
        batch_size = seq_emb.shape[0]
        protein_graph = protein_graph.to(device)
        seq_emb = seq_emb.to(device)
        ligand_graph = ligand_graph.to(device)
        label = label.to(device)
        out = model(protein_graph, seq_emb, ligand_graph)
        loss = loss_fn(out.view(-1), label.view(-1))
        #print(label)
        loss.backward()
        optimizer.step()
        meter.update(loss.detach().cpu().item(), batch_size)
    return meter.avg
        
@torch.no_grad()
def eval_one_epoch(model, loader, device, evaluator, loss_fn = None):
    model.eval()
    if loss_fn is not None:
        meter = AverageMeter()
    Y = []
    P = []
    for batch in tqdm(loader, total = len(loader)):
        protein_graph, seq_emb, ligand_graph, label = batch
        batch_size = torch.max(protein_graph.batch) + 1
        protein_graph = protein_graph.to(device)
        seq_emb = seq_emb.to(device)
        ligand_graph = ligand_graph.to(device)
        Y += list(label.view(-1).numpy())
        pred = model(protein_graph, seq_emb, ligand_graph).view(-1)
        if loss_fn is not None:
            batch_size = len(label)
            label = torch.tensor(label).cuda()
            loss = loss_fn(pred, label.view(-1))
            meter.update(loss.detach().cpu().item(), batch_size)
        P += list(pred.cpu().numpy())
    
    Y = np.array(Y)
    P = np.array(P)
    result_dict = evaluator(Y, P)
    if loss_fn is not None:
        result_dict['avg_loss'] = meter.avg
    return result_dict

def train_one_epoch_davis_and_kiba(model, loader, optimizer, loss_fn, device, evaluator = None):
    model.train()
    meter = AverageMeter()
    for batch in tqdm(loader, total = len(loader)): 
        optimizer.zero_grad()
        protein_graph, seq_emb, ligand_graph, ligand_mp, label = batch
        batch_size = seq_emb.shape[0]
        protein_graph = protein_graph.to(device)
        seq_emb = seq_emb.to(device)
        ligand_graph = ligand_graph.to(device) 
        ligand_mp = ligand_mp.to(device)
        label = label.to(device)
        out = model(protein_graph, seq_emb, ligand_graph, ligand_mp)
        loss = loss_fn(out.view(-1), label.view(-1))
        loss.backward()
        optimizer.step()
        meter.update(loss.detach().cpu().item(), batch_size)
    return meter.avg
        
@torch.no_grad()
def eval_one_epoch_davis_and_kiba(model, loader, device, evaluator, loss_fn = None):
    model.eval()
    if loss_fn is not None:
        meter = AverageMeter()
    Y = []
    P = []
    for batch in tqdm(loader, total = len(loader)):
        protein_graph, seq_emb, ligand_graph, ligand_mp, label = batch
        batch_size = torch.max(protein_graph.batch) + 1
        protein_graph = protein_graph.to(device)
        seq_emb = seq_emb.to(device)
        ligand_graph = ligand_graph.to(device)
        ligand_mp = ligand_mp.to(device)
        Y += list(label.view(-1).numpy())
        pred = model(protein_graph, seq_emb, ligand_graph, ligand_mp).view(-1)
        if loss_fn is not None:
            batch_size = len(label)
            label = torch.tensor(label).cuda()
            loss = loss_fn(pred, label.view(-1))
            meter.update(loss.detach().cpu().item(), batch_size)
        P += list(pred.cpu().numpy())
    
    Y = np.array(Y)
    P = np.array(P)
    result_dict = evaluator(Y, P)
    if loss_fn is not None:
        result_dict['avg_loss'] = meter.avg
    return result_dict