import torch
import math
import numpy as np
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = "cpu"
    
# add interaction to dataset
def add_int(data,p, XM=False):
    X = data[:,:p]
    A = data[:,p]
    XA = (X.T * A).T
    if XM:
        M = data[:,p+1:-1]
        for i in range(p):
            XMi = (M.T * X[:,i]).T
            if not i: XM = XMi[:,:]
            else: XM = np.concatenate(XM,XMi,axis=1)
        return np.concatenate(( data[:,:p+1], XA, data[:,p+1:-1], XM, data[:,-1]), axis=1)
    return np.concatenate(( data[:,:p+1], XA, data[:,p+1:]), axis=1)

# mse between estimated data and data
def mse(B,X):
    return np.mean(np.square(np.matmul(B.T, X.T).T - X))

# tolerance level that optimizes above mse
def get_opt_tol(X,B,p, res = 1000):
    min_tol = min_err = np.Inf
    for tol in np.linspace(0,0.5,res):
        B_p = np.where(abs(B) > tol, B, 0)
        if not nx.is_directed_acyclic_graph(nx.DiGraph(B_p)):
            continue
        err = mse(B_p,X)
        if err < min_err:
            min_err = err
            min_tol = tol
    return min_tol

# build sub graph from full model
def gen_sub(B, x_vec):
    p = x_vec.shape[-1]
    s = B.shape[0] - 2 - 2*p
    
    # init arrays
    sub = np.zeros([s+2,s+2])
    intercept = np.zeros([s+2])
    
    # add in coefficients and remove zeros
    sub[1:,0] = np.squeeze(B.T[(2*p+1):,p]) +  np.squeeze(B.T[(2*p+1):,(p+1):(2*p+1)].dot(x_vec))
    sub[1:,1:-1] = B.T[(2*p+1):,(2*p+1):(2*p+s+1)]
    intercept[0] = B.T[p,:p].dot(x_vec)
    intercept[1:] = B.T[(2*p+1):,:p].dot(x_vec)
    
    return sub.T, intercept
    


#========================================
# Calculate Constraints and Update Optimizer
#========================================

def fun_h1_B(B):
    '''compute constraint h1(B) value'''
    d = B.shape[0]
    expm_B = matrix_poly(B * B, d)
    h1_B = torch.trace(expm_B) - d
    return h1_B

def fun_h2_B(B,p,s=0,inter=True):
    '''compute constraint h2(B) value'''
    d = B.shape[0]
    # no parents for X
    h2_B_X = torch.sum(torch.abs(B[:, 0:p]))
    # only X parents for A
    h2_B_A = torch.sum(torch.abs(B[:, p]))- torch.sum(torch.abs(B[:p, p]))
    # no parents for interaction
    h2_B_XA = torch.sum(torch.abs(B[:,(p+1):(2*p+1)])) if inter else 0
    h2_B_XM = torch.sum(torch.abs(B[:,(2*p+s+1):(2*p+s+(p*s)+1)])) if (inter and s) else 0
    # (for now) XM cannot be parents of M
    h2_B_M = torch.sum(torch.abs(B[(2*p+s+1):(2*p+s+(p*s)+1),2*p+1:2*p+s+1])) if (inter and s) else 0
    # no descendents for Y
    h2_B_Y = torch.sum(torch.abs(B[(d - 1), :]))-torch.sum(torch.abs(B[(d - 1), :(p+1)]))

    h2_B = h2_B_X + h2_B_A + h2_B_Y + h2_B_XA + h2_B_XM + h2_B_M
    return h2_B
    
def update_optimizer(optimizer, old_lr, c_B, d_B):
    '''related LR to c_B and d_B, whenever c_B and d_B gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4
    
    estimated_lr = old_lr / (math.log10(c_B) + math.log10(d_B) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr
    
    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


#========================================
# VAE utility functions and modules
# Credit to DAG-GNN https://github.com/fishmoon1234/DAG-GNN
#========================================

_EPS = 1e-10
prox_plus = torch.nn.Threshold(0.,0.)

def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
        enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def kl_gaussian(preds):
    """compute the KL loss for Gaussian variables."""
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def nll_gaussian(preds, target, variance, add_const=False):
    """compute the loglikelihood of Gaussian variables."""
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

def preprocess_adj(adj):
    """preprocess the adjacency matrix: adj_A = I-A^T."""
    adj_normalized = (torch.eye(adj.shape[0]).double().to(device) - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_inv(adj):
    """preprocess the adjacency matrix: adj_A_inv = (I-A^T)^(-1)."""
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double().to(device)-adj.transpose(0,1))
    return adj_normalized

def matrix_poly(matrix, d):
    x = torch.eye(d).double().to(device)+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(MLPEncoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double().to(device), requires_grad=True))
        self.factor = factor
        self.Wa = nn.Parameter(torch.zeros(n_out).to(device), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol).to(device))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double().to(device))
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')
        
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        
        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj(adj_A1)
        
        adj_A = torch.eye(adj_A1.size()[0]).double().to(device)
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa
        
        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class MLPDecoder(nn.Module):
    """MLP decoder module."""
    
    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()
        
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)
        
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        
        self.dropout_prob = do_prob
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):
        
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_inv(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa
        
        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)
        
        return mat_z, out, adj_A_tilt