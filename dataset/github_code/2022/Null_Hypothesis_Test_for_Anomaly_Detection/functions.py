#!/usr/bin/env python
# coding: utf-8


# General functions

import numpy as np

#Useful to deal with binned data
def bin_assigner(bins,val):
  if val < bins[0] or val >= bins[-1]:
    bin_index=-1
  else:
    for m in range(len(bins)-1):
      if bins[m] <= val and val < bins[m+1]:
        bin_index = m
        break
  return bin_index

# for jet clustering and analysis

def plaindistance(j1, j2):
        return np.sqrt((j1.eta-j2.eta)*(j1.eta-j2.eta)+(j1.phi-j2.phi)*(j1.phi-j2.phi))

# nsubjetiness calculation
def nsubjet(jeteval, Rtouse, N):
        d0 = 0.0
        tauN = 0.0
        if(len(jeteval.constituents_array()) < N):
                return tauN
        else:
                sequence = cluster(jeteval.constituents_array(), R=float(Rtouse/N), p=1)
                jets = sequence.exclusive_jets(int(N))
                for constindex, constituent in enumerate(jeteval.constituents()):
                        d0 += constituent.pt
                        tauMin = Rtouse
                        for subjetindex, subjet in enumerate(jets):
                                if tauMin > plaindistance(subjet,constituent):
                                        tauMin = plaindistance(subjet, constituent)
                        tauN += tauMin*constituent.pt
                tauN = float(tauN/(d0*Rtouse))
                return tauN


# Mutual Information

def mutual_info(pxy,px,py):# I'll give it a shape of (dx,dy), (dx,), (dy,)
	MI = 0.0
	dx = len(px)
	dy = len(py)
	if pxy.shape[0]!=dx or pxy.shape[1]!=dy:
		return "Wrong shapes"
	else:
		MI=np.sum(list(map(lambda indx: list(map(lambda indy: pxy[indx,indy]*(np.log(pxy[indx,indy])-np.log(px[indx])-np.log(py[indy])) if pxy[indx,indy]>0.0 else 0.0,range(dy))),range(dx))))
		return MI

#


# Here comes the NN implementation for CWoLA

import torch
from torch import nn
from tqdm import tqdm
from torch.nn.modules import Module

class xyDataset(torch.utils.data.Dataset):
    """
    Joins the x and y into a dataset, so that it can be used by the pythorch syntax.
    """

    def __init__(self, x, y,w):
        self.x = torch.tensor(x).to(torch.float)
        self.y = torch.tensor(y).to(torch.float)
        self.w = torch.tensor(w).to(torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.x[idx], self.y[idx],self.w[idx]]
        return sample


# Define model


class NeuralNetwork(nn.Module):
    def __init__(self,dim_input=6,layers_data=[(6, nn.ReLU()), (1, nn.Sigmoid())]): # example of layers_data=[(layer1, nn.ReLU()), (layer2, nn.ReLU()), (output_size, nn.Sigmoid())]
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.input_size = dim_input  # Can be useful later ...
        input_size = dim_input
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

        device = "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, x):
        output = self.flatten(x)
        for layer in self.layers:
            output = layer(output)
        # output = self.linear_relu_stack(x)


        return output

    def loss_function(self,t,y,w):
        loss_fn = nn.BCELoss(weight=w)
        return loss_fn(t,y)

    def reset_weights(self):
        for m in self.layers:#linear_relu_stack:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def Train(self,train, batch_size=1024,epochs=3,learning_rate=1e-3,num_workers=0):
        # Define the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=True, num_workers=num_workers)
        # define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        size = len(train)
        avg_loss = []
        epoch_list=[]
        # self.train()

        for epoch in tqdm(range(0, epochs + 1), ncols = 100):
            if epoch > 0:  # test untrained net first
                self.train() 
                train_loss = 0.0
                for batch, (X, y, w) in enumerate(train_loader):
                    X, y, w  = X.to(device), y.to(device), w.to(device)

                    # Compute prediction error
                    pred = self(X)
                    loss = self.loss_function(pred,y,w)

                    train_loss += loss
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if batch % 100 == 0:
                        # loss, current = loss.item(), batch * len(X)
                        pass
                if epoch%5==0:
                    print('----> Learning rate: ', optimizer.param_groups[0]['lr'])
                    print(f'----> Epoch: {epoch} Average loss: {train_loss/ len(train_loader.dataset):.5f} ') 
                    avg_loss.append(train_loss/ len(train_loader.dataset))
                    epoch_list.append(epoch)