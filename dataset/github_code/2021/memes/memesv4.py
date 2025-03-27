import numpy as np
from math import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle

DEVICE = "cpu"

MSG_HID = 30
LOGIT_HID = 16
GLOBAL_HID = 16
GAIN = 3
LOGIT_GAIN = 1

REP_RATE = 0.1
MUTATION_SCALE = 0.2
MUTATION_PROBA = 0.001 # Overridden by experiment parameters

def adapt(x, Htarg):
    z = x
    mz,_ = torch.max(z,0)
    z = z - mz
    
    for i in range(20):
        p = torch.exp(z)
        p = p/torch.sum(p,0).unsqueeze(0)
        H = -torch.sum(p*torch.log(p+1e-16),0).unsqueeze(0)
        w = 1.0 + 0.1 * (H-Htarg)/Htarg
        z = z * w
    
    return z

class LogitNet(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        
        self.l1 = nn.Linear(MSG_HID + GLOBAL_HID, LOGIT_HID)
        nn.init.orthogonal_(self.l1.weight, gain=sigma)
        self.l2 = nn.Linear(LOGIT_HID, LOGIT_HID)
        nn.init.orthogonal_(self.l2.weight, gain=sigma)
        self.l3 = nn.Linear(LOGIT_HID, 1)
        nn.init.orthogonal_(self.l3.weight, gain=sigma)
        
    def forward(self, msg, hid):
        z = torch.cat([msg, hid.expand(msg.size(0), GLOBAL_HID)], 1)
        
        z = F.relu(self.l1(z))
        z = F.relu(self.l2(z))
        z = self.l3(z)
        
        return z

class MessageNet(nn.Module):
    def __init__(self, sigma, skip):
        super().__init__()
        
        self.l1 = nn.Linear(MSG_HID + GLOBAL_HID, LOGIT_HID)
        nn.init.orthogonal_(self.l1.weight, gain=sigma)
        self.l2 = nn.Linear(LOGIT_HID, LOGIT_HID)
        nn.init.orthogonal_(self.l2.weight, gain=sigma)
        self.l3 = nn.Linear(LOGIT_HID, MSG_HID)
        nn.init.orthogonal_(self.l3.weight, gain=sigma)
        self.skip = skip
        
    def forward(self, msg, hid):
        z = torch.cat([msg, hid.expand(msg.size(0), GLOBAL_HID)], 1)
        
        z = F.relu(self.l1(z))
        z = F.relu(self.l2(z))
        z = self.l3(z)
        
        if self.skip:
            p = torch.sigmoid(GAIN * z + msg)
        else:
            p = torch.sigmoid(GAIN * z)
        
        return 2*torch.le(torch.rand_like(p), p).float()-1
    
class Net(nn.Module):
    def __init__(self, sigma, skip):
        super().__init__()
        
        self.htoh = nn.LSTMCell(MSG_HID, GLOBAL_HID)
        self.logit = LogitNet(sigma)
        self.msg = MessageNet(sigma, skip)
    
    def setup(self, MEM_SIZE = 120):
        self.age = 0
        self.kids_other = 0
        self.kids_self = 0
        
        self.memory = torch.zeros(MEM_SIZE, MSG_HID).to(DEVICE)
        self.sources = torch.zeros(MEM_SIZE).long().to(DEVICE)
        self.totals = torch.zeros(MEM_SIZE).to(DEVICE)
        self.counts = torch.zeros(MEM_SIZE).to(DEVICE)
        
        self.message = 2*torch.randint(0,2, (1, MSG_HID)).to(DEVICE).float()-1
        self.hid = torch.zeros(1, GLOBAL_HID).to(DEVICE)
        self.cell = torch.zeros(1, GLOBAL_HID).to(DEVICE)
    
    def add_message(self, msg, src):
        self.memory = torch.cat([self.memory[1:], msg], 0)
        self.sources = torch.cat([self.sources[1:], torch.LongTensor([src]).to(DEVICE)], 0)
        self.totals = torch.cat([self.totals[1:], torch.zeros(1,).to(DEVICE)], 0)
        self.counts = torch.cat([self.counts[1:], torch.zeros(1,).to(DEVICE)], 0)
    
    def mutate(self):
        for p in self.parameters():
            g = torch.le(torch.rand_like(p.data),MUTATION_PROBA).float()
            p.data = (1-g)*p.data + g*(0.99 * p.data + MUTATION_SCALE * torch.randn_like(p.data).to(DEVICE))
    
    def flat_params(self):
        fparams = []
        for p in self.parameters():
            fparams.append(p.view(-1))
        
        return torch.cat(fparams,0).cpu().detach().numpy()
    
    def replicate(self, parent1, parent2):
        for p1,p2,p3 in zip(self.parameters(), parent1.parameters(), parent2.parameters()):
            p1.data = p3.data.detach().clone()
        
        self.mutate()
        
    def step(self):
        mem = self.memory
        
        logits = torch.clamp(self.logit(mem, self.hid) * LOGIT_GAIN,-30.0,30.0) # For numerical stability
        
        w = torch.exp(logits).squeeze()
        w = w/torch.sum(w)
        
        self.totals += w
        self.counts += 1
        
        mixed_msg = torch.sum(w.unsqueeze(1)*mem,0).unsqueeze(0)
        
        self.message = self.msg(mixed_msg, self.hid).detach()
        self.hid, self.cell = self.htoh(mixed_msg, (self.hid, self.cell))
        self.age += 1

default_params = {
    "sigma": 4,
    "RES": 32,
    "mutation": 0.001,
    "select": True,
    "uniform_init": False,
    "output_dir": "baseline",
    "seed": 12345,
    "runlength": 20000,
    "skip": False
}

def merge_keys(x,y):
    z = y    
    for k in x.keys():
        z[k] = x[k]    
    return z

def do_experiment(params):
    try:
        os.mkdir(params["output_dir"])
    except:
        pass
    
    global MUTATION_PROBA
    
    MUTATION_PROBA = params["mutation"]
        
    np.random.seed(50000)
    
    net = Net(0, False)
    FPSIZE = net.flat_params().shape[0]       
    proj = np.random.randn(FPSIZE, MSG_HID)
    
    np.random.seed(params["seed"])
    
    params = merge_keys(params, default_params)
    RES = params["RES"]
    NNETS = RES*RES
    
    net0 = Net(params["sigma"], params["skip"]).to(DEVICE)
    nets = [Net(params["sigma"], params["skip"]).to(DEVICE) for i in range(NNETS)]
    
    if params["uniform_init"]:
        for n in nets:
            n.load_state_dict(net0.state_dict())
    
    link = np.zeros((NNETS, NNETS))

    adj = []

    for i in range(-2,3):
        for j in range(-2,3):
            if i!=0 or j!=0:
                adj.append([i,j])
            
    for i in range(RES):
        for j in range(RES):
            k = i+j*RES
            
            for a in adj:
                km = (i+RES+a[0])%RES + ((j+RES+a[1])%RES)*RES
                link[k, km] = 1
                
    for n in nets:
        n.setup()
        
    for l in range(5):
        for i in range(len(nets)):
            for j in range(len(nets)):
                if link[i,j]:
                    nets[i].add_message(nets[j].message, j)

    meme_set = []
    gene_set = []

    with torch.no_grad():
        for t in range(params["run_length"]):
            meme_dict = {}
            gene_dict = {}
            
            messages = []

            for i in range(len(nets)):
                for j in range(len(nets)):
                    if link[i,j]:
                        nets[i].add_message(nets[j].message, j)
                        
            for n in nets:
                n.step()
            
            for i in range(len(nets)):
                m = nets[i].message.cpu().detach().numpy()
                
                mstr = ""
                for j in range(m.shape[1]):
                    if m[0,j]<0:
                        mstr += "0"
                    else:
                        mstr += "1"
                
                if mstr in meme_dict:
                    meme_dict[mstr] += 1
                else:
                    meme_dict[mstr] = 1
                
                p = np.matmul(nets[i].flat_params(), proj)
                
                gstr = ""
                for j in range(p.shape[0]):
                    if p[j]<0:
                        gstr += "0"
                    else:
                        gstr += "1"
                
                if gstr in gene_dict:
                    gene_dict[gstr] += 1
                else:
                    gene_dict[gstr] = 1
                
                mt = torch.min(nets[i].counts)
                if np.random.rand()<REP_RATE and mt>0 and torch.sum(nets[i].totals)>0:
                    nets[i].kids_self += 1
                    
                    if params["select"]:
                        p = (nets[i].totals/nets[i].counts).cpu().detach().numpy()
                        p = p/np.sum(p)
                        j = np.random.choice( np.arange(nets[i].totals.shape[0]), p=p)
                    else:
                        j = np.random.randint(nets[i].totals.shape[0])
                        
                    src = nets[i].sources[j].cpu().detach().item()
                    
                    k = np.random.randint(len(adj))
                    k2 = 0
                    for j in range(len(nets)):
                        if link[i,j]:
                            if k2==k:
                                nets[j].replicate(nets[i], nets[src])
                                nets[j].setup()
                                for m in range(5):
                                    for l in range(len(nets)):
                                        if link[j,l]:
                                            nets[j].add_message(nets[l].message, l)
                                k2 += 1
                            else:
                                k2 += 1         
            
            meme_set.append(meme_dict)
            gene_set.append(gene_dict)
            
            if t%500 == 499:
                pickle.dump(meme_set, open("%s/memes-%.6d.pkl" % (params["output_dir"], params["seed"]),"wb"))
                pickle.dump(gene_set, open("%s/genes-%.6d.pkl" % (params["output_dir"], params["seed"]),"wb"))
