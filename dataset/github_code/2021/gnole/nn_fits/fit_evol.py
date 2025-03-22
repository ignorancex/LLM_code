#!/usr/bin/env python
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

alphas=0.118
ca=3.0
nf=5.0
cf=4.0/3.0
b0 = (11*ca - 2*nf)/(12*math.pi)
b1 = (17*ca*ca - 5*ca*nf - 3*cf*nf)/(24*math.pi**2)

def tscale(L):
    """the t value corresponding to a given ln kt scale"""
    return (2*alphas*alphas*b0*b1*L + (b0 + alphas*b1 - 2*alphas*b0*b0*L)*\
	    np.log(1 - 2*alphas*b0*L))/(4.*b0*b0*(-1 + 2*alphas*b0*L)*math.pi)

def ln_kt_grid(npoints):
    """get a grid of (L, t(L)) points"""
    Lmax = 1.0/(2.0*alphas*b0)
    # L = np.linspace(0, Lmax, npoints)
    L = np.arange(0, Lmax, Lmax/npoints)[1:]
    t = tscale(L)
    return t, L

npts =30000
EPOCH=50000
maxt =20.0
# npts =2000
# EPOCH=2000
BATCH_SIZE = 32
xnp, ynp = ln_kt_grid(npts)
data = np.column_stack((xnp,ynp))
data = data[(data[:,0]<maxt),:]

x = Variable(torch.unsqueeze(torch.from_numpy(np.log(data[:,0])).float(),dim=1))
y = Variable(torch.unsqueeze(torch.from_numpy(np.log(data[:,1])).float(),dim=1))
# data.sort(axis=0)
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')

torch.manual_seed(1)    # reproducible
net = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 1),
    )


optimizer = torch.optim.Adam(net.parameters(), lr=1e-6) #1e-5
loss_func = torch.nn.MSELoss()

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=0)
net.to(dev)

# start training
for epoch in range(EPOCH):
    batch_loss=0.0
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        prediction = net(b_x.to(dev))     # input x and predict based on x

        loss = loss_func(prediction, b_y.to(dev))     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        batch_loss=max(batch_loss,loss)
    if (epoch%50==0):
        l=len(str(EPOCH))
        print('[%s/%i] loss: %9.6f' % (str(epoch).rjust(l), EPOCH, batch_loss))

#output the model
ex=torch.rand(1,1)
traced_script_module=torch.jit.trace(net,ex.to(dev))
ptfn='lnkt_model_as%1.3f_ca%1.1f_cf%1.4f_nf%i.pt' % (alphas, ca, cf, nf)
traced_script_module.save(ptfn)
# plot test
plt.plot(data[:,0],data[:,1],label='data') # data[:,1]*norm
xpred=Variable(torch.unsqueeze(torch.linspace(-14, 2.0, 500), dim=1))
prediction = net(xpred.to(dev))     # input x and predict based on x
plt.plot(np.exp(xpred.data.cpu().numpy()),np.exp(prediction.data.cpu().numpy()),label='pred',ls='--')
plt.legend()
plt.show()

plt.plot(np.log(data[:,0]),np.log(data[:,1]),label='data') # data[:,1]*norm
plt.plot(xpred.data.cpu().numpy(),prediction.data.cpu().numpy(),label='pred',ls='--')
plt.legend()
plt.show()
