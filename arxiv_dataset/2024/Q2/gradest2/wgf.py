import torch
from matplotlib import pyplot as plt
from core.util import comp_dist, comp_median, kernel_comp, dKernel_comp

# psicon = lambda d: torch.exp(d - 1)
psicon = lambda d: d**2/2 + d
rbfkernel = lambda x, y, sigma: kernel_comp(x, y, sigma)

def obj(W, b, xp, xq, psicon, kpx=None, kqx=None):
    A = torch.mean(kpx * (torch.matmul(W, xp.T) + b), 1, keepdim=True)
    B = torch.mean(kqx * psicon( (torch.matmul(W, xq.T) + b)), 1, keepdim=True)
    return torch.mean(- A + B, 0)

def gradest(net, xp, xq, x, sigma, optimizer, kernel = rbfkernel, batch_size = 256, nepochs = 100, penalty = None):
    
    d = xp.shape[1]

    kpx = kernel(x, xp, sigma)
    kqx = kernel(x, xq, sigma)
    
    # create the neural network
    train_loader = torch.utils.data.DataLoader( 
                torch.utils.data.TensorDataset(xp, xq, kpx.T, kqx.T), batch_size=batch_size, shuffle=True)
    
    # start the training loop
    for epoch in range(nepochs):
        # iterate over the data
        for i, data in enumerate(train_loader):
            # get the batch
            xpi, xqi, kpxi, kqxi = data
            kpxi = kpxi.T; kqxi = kqxi.T
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predicted = net(x)
            Wx = predicted[:, :d]
            bx = predicted[:, d:]
            
            output = obj(Wx, bx, xpi, xqi, psicon, kpxi, kqxi)
            
            if penalty is not None:
                output = output + penalty(x)
            
            output.backward()
            optimizer.step()

            # # print statistics
            # if epoch % 100 == 0:
            #     print('[%d, %5d] loss: %f' %(epoch, i, output.item()))
                
    return net

def MMD_flow(zp, zq, kernel = rbfkernel, sigma = None):
    zq.requires_grad = True
    kqq = kernel(zq, zq, sigma)
    kpq = kernel(zp, zq, sigma)
    
    MMDobj = torch.mean(kqq) - 2*torch.mean(kpq)
    
    # compute the gradient with respect to xq using automatic differentiation
    grad = torch.autograd.grad(-MMDobj, zq)[0]
    return grad * 1000000
    
    
    
    
    
