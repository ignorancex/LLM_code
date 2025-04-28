# %%

import matplotlib.pyplot as plt
import torch
from numpy import *
from core.util import svm
from core.util import comp_dist, kernel_comp
import classif
from wgf_da import WGF_DomainAdaptation

from IPython import display

import pylab as pl
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def kernel(x,y,sigma):
    # # the first till the second last, use gaussian kernel
    x1 = x[:, :-1]
    y1 = y[:, :-1]
    k1 = kernel_comp(x1, y1, sigma)
    
    # the last dimension, use delta kernel
    x2 = x[:, -1:]
    y2 = y[:, -1:]
    k2 = x2 - y2.T
    k2 = (k2 == 0)
    
    return k1*k2
    # return kernel_comp(x, y, sigma)

# %%
seed = 123
torch.manual_seed(seed)
#fix numpy seed
random.seed(seed)
results = {}

# loop over all pairs of datasets
ds_list = ['amazon', 'dslr', 'webcam', 'caltech']

for ds_s in ds_list:
    for ds_t in ds_list:
        
        if ds_s == ds_t:
            continue
            
        from scipy.io import loadmat
        mat = loadmat('decaf6/' + ds_t + '_decaf.mat')
        Xp = mat['feas']
        yp = mat['labels'].flatten()

        Xp = Xp - mean(Xp, axis=0)
        # do PCA on Xp
        from sklearn.decomposition import PCA
        pca = PCA(n_components=100)
        pca.fit(Xp)
        Xp = pca.transform(Xp)
        Xp = Xp / 100

        Xp_tor = torch.tensor(Xp, dtype=torch.float32).to(device)

        from scipy.io import loadmat
        mat = loadmat('decaf6/' + ds_s + '_decaf.mat')
        Xq = mat['feas']
        yq = mat['labels'].flatten()
        Xq = Xq - mean(Xq, axis=0)
        # do PCA on Xq
        Xq = pca.transform(Xq)
        Xq = Xq / 100
        Xq_tor = torch.tensor(Xq, dtype=torch.float32).to(device)

        gamma = 1/comp_dist(Xq_tor, Xq_tor).flatten().median().item()
        yt, acc_base = svm(Xq_tor.cpu(), yq, Xp_tor.cpu(), yp, gamma = gamma)


        yphat, Xq_T, Xq_traj = WGF_DomainAdaptation(Xp_tor, yp, Xq_tor, yq, kernel, nepoch = 5, VGD_batchsize = 500, device = device)

        # plot the trajectory of Xq_N

        plt.figure(figsize=(10,10), num=999, clear=True)

        for i in range(Xq_tor.shape[0]):
            if yq[i] == 1:
                xi_traj = torch.stack([Xq_traj[t][i, :] for t in range(len(Xq_traj))])
                plt.plot(xi_traj[:, 0].cpu(), xi_traj[:, 1].cpu(), 'b', alpha = 1)
            elif yq[i] == 2:
                xi_traj = torch.stack([Xq_traj[t][i, :] for t in range(len(Xq_traj))])
                plt.plot(xi_traj[:, 0].cpu(), xi_traj[:, 1].cpu(), 'r', alpha = 1)
            elif yq[i] == 3:
                xi_traj = torch.stack([Xq_traj[t][i, :] for t in range(len(Xq_traj))])
                plt.plot(xi_traj[:, 0].cpu(), xi_traj[:, 1].cpu(), 'g', alpha = 1)
                

        # compute accuracy
        acc_proposed = mean(yp==yphat)
        print("accuracy:", acc_proposed)

        gamma = 1/comp_dist(Xq_T, Xq_T).flatten().median().item()
        yt, acc_svm = svm(Xq_T.cpu(), yq, Xp, yp, gamma = gamma)

        import ot
        import jdot
        #from sklearn import datasets
        import classif

        reg=1e1
        itermax=10
        # gamma=0.1#
        # gamma = 1/(2*sigma.item()**2)
        gamma = 1/comp_dist(Xp_tor, Xp_tor).flatten().median().item()

        YY,Yb=classif.get_label_matrix(yq)
        YYT,Yb=classif.get_label_matrix(yp)

        clf_jdot,dic= jdot.jdot_svm(Xq, YY, Xp, ytest=[], ytest2=YYT, gamma_g=gamma,numIterBCD = itermax, alpha=.1, lambd=reg,ktype='linear')#,method='sinkhorn',reg=0.01)

        def predict_test(clf,gamma,Xapp,Xtest):
            Kx=classif.rbf_kernel(Xtest,Xapp,gamma=gamma)
            return clf.predict(Kx) 

        ypred=predict_test(clf_jdot,gamma,Xp,Xp)
        TBR= mean(yp==(ypred.argmax(1))+1)
        # accuracy
        print("accuracy:", TBR)
        
        # save results to a dictionary
        results[ds_s + '->' + ds_t] = {'acc_base': acc_base, 'acc_proposed': acc_proposed, 'acc_svm':acc_svm, 'acc_jdot': TBR}

# %% save results to file

torch.save(results, 'results_decaf6.pt')
# %%
