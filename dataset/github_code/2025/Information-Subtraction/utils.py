import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm


class Z_RNN(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 32, num_layers = 1, output_size = 2):
        super(Z_RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,     # rnn hidden unit
            num_layers = num_layers,       # number of rnn layer
            batch_first = True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        # print(r_out.shape)
        # outs = []    # save all predictions
        # for time_step in range(r_out.size(1)):    # calculate output for each time step
        #     outs.append(self.out(r_out[:, time_step, :]))

        # return torch.stack(outs, dim=1), h_state

        outs = self.out(r_out)
        return outs, h_state

class Mine(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output
    
class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
    
def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, x_dim, y_dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(y_dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores


class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, x_dim, y_dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(x_dim+y_dim, hidden_dim, 1, layers, activation)

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()


"""Main training loop that estimates time-varying MI."""
# Ground truth rho is only used by conditional critic

estimator = 'smile'
clip = 5.0

CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic,
}

def smile(X, Y, size=100):


    X = (X - X.min()) / (X.max() - X.min())
    Y = (Y - Y.min()) / (Y.max() - Y.min())

    data_params = {
        'x_dim': X.shape[1],
        'y_dim': Y.shape[1],
        'batch_size': 1000,
        'cubic': None
    }

    critic_params = {
        'x_dim': X.shape[1],
        'y_dim': Y.shape[1],
        'layers': 2,
        'embed_dim': 32,
        'hidden_dim': 256,
        'activation': 'relu',
    }

    opt_params = {
        'iterations': 20000,
        'learning_rate': 5e-4,
    }

    estimator = 'smile'
    clip = 5.0

    CRITICS = {
        'separable': SeparableCritic,
        'concat': ConcatCritic,
    }
    critic = CRITICS['separable'](rho=None, **critic_params).cuda()
    opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])

    estimates = []

    for i in tqdm(range(opt_params['iterations'])):
        opt_crit.zero_grad()
        idx = np.random.choice(range(X.shape[0]), size=size, replace=False)
        x_samples = X[idx]
        x_samples = torch.autograd.Variable(torch.FloatTensor(x_samples)).cuda()
        y_samples = Y[idx]
        y_samples = torch.autograd.Variable(torch.FloatTensor(y_samples)).cuda()
        x = x_samples
        y = y_samples

        x, y = x.cuda(), y.cuda()
        scores = critic(x, y)
        if clip is not None:
            f_ = torch.clamp(scores, -clip, clip)
        else:
            f_ = scores

        batch_size = f_.size(0)
        logsumexp = torch.logsumexp(
            f_ - torch.diag(np.inf * torch.ones(batch_size).to('cuda')), dim=(0, 1))
        try:
            if len((0, 1)) == 1:
                num_elem = batch_size - 1.
            else:
                num_elem = batch_size * (batch_size - 1.)
        except ValueError:
            num_elem = batch_size - 1
        z = logsumexp - torch.log(torch.tensor(num_elem)).to('cuda')
        dv = scores.diag().mean() - z
        """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
        f_diag = scores.diag()
        first_term = -F.softplus(-f_diag).mean()
        n = scores.size(0)
        second_term = (torch.sum(F.softplus(scores)) -
                    torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
        js = first_term - second_term

        with torch.no_grad():
            dv_js = dv - js
        mi = js + dv_js
        loss = -mi

        loss.backward()
        opt_crit.step()

        mi = mi.detach().cpu().numpy()
        estimates.append(mi)

    return np.array(estimates)

def mine(X, Y):
    x_dim = X.shape[1]
    y_dim = Y.shape[1]

    X = (X - X.min() ) / (X.max() - X.min() )
    Y = (Y - Y.min() ) / (Y.max() - Y.min() )

    X = torch.autograd.Variable(torch.FloatTensor(X)).cuda()
    Y = torch.autograd.Variable(torch.FloatTensor(Y)).cuda()

    mine_net = Mine(input_size = x_dim+y_dim, hidden_size = 512).cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3, weight_decay = 1e-5)
    batch_size=128
    iter_num=int(3e+4)

    result = list()
    ma_et = 1.
    ma_rate=0.01
    for i in tqdm(range(iter_num)):
        # batch is a tuple of (joint, marginal)
        idx = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        x = X[idx]
        y = Y[idx]

        idx = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        marginal_Y = Y[idx]

        t = mine_net( torch.cat((y, x), axis = 1) )
        marginal_t = mine_net( torch.cat((marginal_Y, x), axis = 1) )

        et = torch.exp(marginal_t)
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
        
        # unbiasing use moving average
        loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
        # use biased estimator
    #     loss = - mi_lb
        
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        # if (i+1)%(10)==0:
        result.append(mi_lb.detach().cpu().numpy())
    return result


def return_smile_loss(scores, clip=5.0):
    if clip is not None:
        f_ = torch.clamp(scores, -clip, clip)
    else:
        f_ = scores

    batch_size = f_.size(0)
    logsumexp = torch.logsumexp(
        f_ - torch.diag(np.inf * torch.ones(batch_size).to('cuda')), dim=(0, 1))
    try:
        if len((0, 1)) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    z = logsumexp - torch.log(torch.tensor(num_elem)).to('cuda')
    dv = scores.diag().mean() - z
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = scores.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = scores.size(0)
    second_term = (torch.sum(F.softplus(scores)) -
                torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    js = first_term - second_term

    with torch.no_grad():
        dv_js = dv - js
    mi = js + dv_js
    mine_loss = -mi
    return mine_loss

class Regressor(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 32, output_size = 1):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, output_size)
                                )
        
    def forward(self, input):
        return self.seq(input)
    
def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]