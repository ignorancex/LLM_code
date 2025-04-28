import sklearn
from scipy.spatial.distance import cdist
import torch
from core.util import comp_dist
from numpy import *
import torch.optim as optim
import classif
import ot
from wgf import gradest, MMD_flow
from core.nn import NPnet

def OTYphat(xq, yq, xp, Chinge = None, lmb = 1e1):
    yyq,Yb=classif.get_label_matrix(yq)
    
    wa=torch.ones((xq.shape[0],))/xq.shape[0]
    wb=torch.ones((xp.shape[0],))/xp.shape[0]
    C0=torch.tensor(cdist(xq.cpu(), xp.cpu(),metric='sqeuclidean'))
    
    if Chinge is None:
        Chinge=torch.zeros_like(torch.tensor(C0))
        
    C=.1*C0+Chinge
    G = ot.emd(wa.cpu().numpy(),wb.cpu().numpy(),C.cpu().numpy())
    # Kt=sklearn.metrics.pairwise.rbf_kernel(Xp,gamma=gam)
    Kt = sklearn.metrics.pairwise.linear_kernel(xp)
    Yst=xp.shape[0]*G.T.dot((yyq+1)/2.)
    g = classif.SVMClassifier(lmb)
    
    g.fit(Kt,Yst)
    ypred=g.predict(Kt)
    yphat = (ypred.argmax(1))+1
    Chinge=classif.loss_hinge(yyq,ypred)
    
    return yphat, Chinge

def concat(xp, yp, xq, yq, device):
    Zp = torch.cat([xp, torch.tensor(yp, dtype=torch.float32, device=device).view(-1,1)], 1)
    # idx = random.choice(Zp.shape[0], xq.shape[0])
    # Zp = Zp[idx, :]

    Zq = torch.cat([xq, torch.tensor(yq, dtype=torch.float32, device=device).view(-1,1)], 1)
    
    return Zp, Zq

def WGF_DomainAdaptation(Xp, yp, Xq, yq, kernel, nepoch = 5, VGD_batchsize = 500, device = 'cpu'):
    yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu())
    Zp, Zq = concat(Xp, yphat, Xq, yq, device)

    idxp = torch.tensor(range(Zp.shape[0]))
    idxq = torch.tensor(range(Zq.shape[0]))
    
    # if two datasets are not the same, resample the small dataset to match the big dataset. 
    if Zp.shape[0] > Zq.shape[0]:
        idxq = random.choice(Zq.shape[0], Zp.shape[0])
    elif Zp.shape[0] < Zq.shape[0]:
        idxp = random.choice(Zp.shape[0], Zq.shape[0])

    train_loader = torch.utils.data.DataLoader( 
                torch.utils.data.TensorDataset(torch.tensor(idxp), 
                torch.tensor(idxq)), batch_size=VGD_batchsize, shuffle=True)
    Xq_traj = [Xq]

    for epoch in range(nepoch):
        for i, data in enumerate(train_loader):
            
            print("iteration:", i, "...")
            idxp, idxq = data
            Zpi = Zp[idxp, :]; Zqi = Zq[idxq, :]
            
            lmbd = 0.0000
            sigma = (.5*comp_dist(Xq, Xp).flatten().median()).sqrt()
                    
            # net = NN_toy(1000, 3).to(device)
            # net = CustomCNN().to(device)
            net = NPnet(Xq.shape[0], Xp.shape[1] + 1).to(device)
            # create the optimizer
            optimizer = optim.Adagrad(net.parameters(), lr=1e-1)

            # Reserse KL flow
            gradnet = gradest(net, Zpi, Zqi, Zq, sigma, optimizer, batch_size=500, nepochs = 500, kernel=kernel)
            
            # #get rid of the bias term
            grad = gradnet(Zq)[:, :Xq.shape[1]].detach()
            
            # grad = MMD_flow(Zp, Zq, kernel = kernel, sigma = sigma)[:, :Xq.shape[1]].detach()
            
            # gradient variational descent
            Xq = Xq + .01*grad
            
            yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu(), Chinge = Chinge)
            Zp, Zq = concat(Xp, yphat, Xq, yq, device)

            Xq_traj.append(Xq)
            
            accuracy = mean(yp==yphat)
            print("accuracy:", accuracy)
                        
        print("done!\n")
    
    return yphat, Xq, Xq_traj