from __future__ import division
from __future__ import print_function

from utils import *
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import random
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Lasso, LinearRegression

if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = "cpu"

def refit(est_graph, data, tol=0.4, alpha=0.03):
    """Function for re-estimating the estimated graph using LASSO.

    Args:
        est_graph: estimated graph
        data: data used to estimate graph
        tol:  threshold for estimating causal skeleton
        alpha: hyper-parameter for LASSO

    Returns:
        W_hat: re-estimated graph
    """
    # get adjacency matrix using tolerance
    datac = np.copy(data)
    A = abs(est_graph.T) >= tol
    # go row by row and run regression
    W_hat = np.zeros_like(est_graph)
    for row in range(A.shape[0]):
        if np.sum(A[row]) == 0: continue
        A_row = np.ravel(A[row])
        if alpha != 0:
            
            W_hat[A_row,row] = Lasso(fit_intercept=False,
                                     alpha=alpha).fit(datac[:,A_row],
                                                      datac[:,row]).coef_
        else:
            W_hat[A_row,row] = LinearRegression(fit_intercept=False).fit(datac[:,A_row],
                                                                         datac[:,row]).coef_
    return W_hat    

#========================================
# Main function
#========================================

def isl(X,s,p,epochs=200,batch_size=64,k_max_iter=1e2,original_lr=3e-3,show=False,seed=123,inter=1,
       inter_XM=0,no_constraints=False):
    """Main algorithm.

    Args:
        X: data of interest
        s: number of potential mediators
        p: number of potential moderators
        batch_size: batch size used for training
        k_max_iter: max number of iterations
        original_lr: starting learning rate
        seed: seed used for the algorithm
        show: if True, the plot of the estimated graph will also be displayed
        inter: if True (or 1), the dimension d is assumed to be 2+s+2*p instead of 2+s+p
        inter_XM: if True (or 1), the dimension d is assumed to be 2+(p+1)*s+p instead of 2+s+p
        no_constraints: if True, no structural constraints will be used in training

    Returns:
        predB: estimated graph
        best_ELBO_loss: best ELBO loss produced over the iterations
        best_NLL_loss: best NLL loss produced over the iterations
        best_MSE_loss: best MSE loss produced over the iterations
    """
    # X is data
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # ----------- Configurations:
    d = ((p+1 if inter_XM else 1)*s)+2+((2 if inter else 1)*p) # The number of variables in data.
    x_dims = 1 # The number of input dimensions: default 1.
    z_dims = d # The number of latent variable dimensions: default the same as variable size.
    p_dims = p # The dimension of confounder node
    epochs = epochs # Number of epochs to train.
    batch_size = batch_size # Number of samples per batch. note: should be divisible by sample size, otherwise throw an error.
    k_max_iter = int(k_max_iter) # The max iteration number for searching parameters.
    original_lr = original_lr  # Initial learning rate.
    encoder_hidden = d^2 # Number of hidden units, adaptive to dimension of nodes (d^2).
    decoder_hidden = d^2 # Number of hidden units, adaptive to dimension of nodes (d^2).
    factor = True # Factor graph model.
    encoder_dropout = 0.0 # Dropout rate (1 - keep probability).
    decoder_dropout = 0.0 # Dropout rate (1 - keep probability).
    tau_B = 0. # Coefficient for L-1 norm of matrix B.
    lambda1 = 0. # Coefficient for DAG constraint h1(B).
    lambda2 = 0. # Coefficient for identification constraint h2(B).
    c_B = 1 # Coefficient for absolute value h1(B).
    d_B = 1 # Coefficient for absolute value h2(B).
    h1_tol = 1e-8 # The tolerance of error of h1(B) to zero.
    h2_tol = 1e-8 # The tolerance of error of h2(B) to zero.
    lr_decay = 200 # After how many epochs to decay LR by a factor of gamma. 
    gamma = 1.0 # LR decay factor.  
    
    # ----------- Training:
    def train(epoch, lambda1, c_B, lambda2, d_B, optimizer, old_lr,inter=True):
        
        nll_train = []
        kl_train = []
        mse_train = []
        encoder.train()
        decoder.train()

        # Update optimizer
        optimizer, lr = update_optimizer(optimizer, old_lr, c_B, d_B)

        for batch_idx, (data, relations) in enumerate(train_loader):

            data, relations = Variable(data).double().to(device), Variable(relations).double().to(device)
            relations = relations.unsqueeze(2) # Reshape data

            optimizer.zero_grad()

            enc_x, logits, origin_B, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send) 
            edges = logits # Logits is of size: [num_sims, z_dims]

            dec_x, output, adj_A_tilt_decoder = decoder(data, edges, d * x_dims, rel_rec, rel_send, origin_B, adj_A_tilt_encoder, Wa)

            if torch.sum(output != output):
                print('nan error\n')

            target = data
            preds = output
            variance = 0.
            
            # Compute constraint functions h1(B) and h2(B)
            h1_B = fun_h1_B(origin_B)
            h2_B = 0 if no_constraints else fun_h2_B(origin_B,p_dims,s if inter_XM else 0,inter)

            # Reconstruction accuracy loss:
            loss_nll = nll_gaussian(preds, target, variance)
            # KL loss:
            loss_kl = kl_gaussian(logits)
            # ELBO loss:
            loss = loss_kl + loss_nll
            # Loss function:
            loss += lambda1 * h1_B + 0.5 * c_B * h1_B * h1_B + lambda2 * h2_B + 0.5 * d_B * h2_B * h2_B + 100. * torch.trace(origin_B * origin_B)

            loss.backward()
            loss = optimizer.step()
            scheduler.step()

            myA.data = stau(myA.data, tau_B * lr)

            if torch.sum(origin_B != origin_B):
                print('nan error\n')

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), origin_B, optimizer, lr
  
    feat_train = torch.FloatTensor(X)

    # Reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    train_loader = DataLoader(train_data, batch_size = batch_size)

    # ----------- Load modules:
    off_diag = np.ones([d, d]) - np.eye(d) # Generate off-diagonal interaction graph
    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype = np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype = np.float64)
    rel_rec = torch.DoubleTensor(rel_rec).to(device)
    rel_send = torch.DoubleTensor(rel_send).to(device)
    adj_A = np.zeros((d, d)) # Add adjacency matrix

    encoder = MLPEncoder(d * x_dims, x_dims, encoder_hidden,
                             int(z_dims), adj_A,
                             batch_size = batch_size,
                             do_prob = encoder_dropout, factor = factor).double().to(device)
    decoder = MLPDecoder(d * x_dims,
                             z_dims, x_dims, encoder,
                             data_variable_size = d,
                             batch_size = batch_size,
                             n_hid=decoder_hidden,
                             do_prob=decoder_dropout).double().to(device)

    # ----------- Set up optimizer:
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = original_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = lr_decay,
                                    gamma = gamma)

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    # ----------- Main:
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    h1_B_new = torch.tensor(1.).to(device)
    h2_B_new = 1 - no_constraints
    h1_B_old = np.inf
    h2_B_old = np.inf
    lr = original_lr

    try:
        for step_k in range(k_max_iter):
            while c_B * d_B < 1e+20:
                for epoch in range(epochs):
                    old_lr = lr 
                    ELBO_loss, NLL_loss, MSE_loss, origin_B, optimizer, lr = train(epoch, lambda1, c_B,
                                                                                   lambda2, d_B, optimizer, old_lr,inter)

                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss

                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # Update parameters
                B_new = origin_B.data.clone()
                if no_constraints:
                    h1_B_new = fun_h1_B(B_new)
                    if h1_B_new.item() > 0.25:
                        c_B *= 10
                    else:
                        break
                else:
                    h1_B_new = fun_h1_B(B_new)
                    h2_B_new = fun_h2_B(B_new,p_dims,inter)
                    if h1_B_new.item() > 0.25 * h1_B_old and h2_B_new > 0.25 * h2_B_old:
                        c_B *= 10
                        d_B *= 10
                    elif h1_B_new.item() > 0.25 * h1_B_old and h2_B_new < 0.25 * h2_B_old:
                        c_B *= 10
                    elif h1_B_new.item() < 0.25 * h1_B_old and h2_B_new > 0.25 * h2_B_old:
                        d_B *= 10
                    else:
                        break

            # Update parameters    
            h1_B_old = h1_B_new.item()
            h2_B_old = h2_B_new
            lambda1 += c_B * h1_B_new.item()
            lambda2 += 0 if no_constraints else d_B * h2_B_new

            if h1_B_new.item() <= h1_tol and h2_B_new <= h2_tol:
                break
                
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    predB = np.matrix(origin_B.data.clone().to('cpu').numpy())
    if show:
        plt.matshow(predB.T, cmap = 'bwr')
        fig1 = plt.gcf()
        plt.colorbar()
        plt.show()
    
    return predB, best_ELBO_loss, best_NLL_loss, best_MSE_loss