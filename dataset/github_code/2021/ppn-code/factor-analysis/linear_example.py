import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy import save
from scipy.stats import chi2
from torch.autograd import Variable
import random

random.seed(123456)

def ppca(x, latent_dim):
    N = x.shape[0]
    input_dim = x.shape[1]

    if torch.is_tensor(x):
        x = x.detach().numpy()

    x_mean = np.mean(x, axis=0)

    x_cent = x - x_mean

    S = np.zeros((input_dim, input_dim))

    for i in range(N):
        S = S + np.outer(x_cent[i, :].T, x_cent[i, :])

    S = S/N
    sigma2 = 1

    diff = 1
    W = np.random.randn(input_dim, latent_dim)
    W_old = W
    iter = 0
    while diff > 1e-5:
        iter += 1
        M = W.T @ W + (sigma2 + 1e-7) * np.eye(latent_dim)
        M_inv = np.linalg.inv(M)
        z = (M_inv @W.T @ x_cent.T).T
        temp = S @ W
        temp_inv = (sigma2 + 1e-7) * np.eye(latent_dim) + M_inv @ W.T @ S @ W
        W = np.matmul(temp, np.linalg.inv(temp_inv))
        sigma2 = 1/input_dim * np.sum(np.diag(S - S @ W_old @ M_inv @ W.T))
        diff = np.sum(np.power(W - W_old, 2))
        W_old = W

    return z, M_inv, W, sigma2

def ppca_est(x, ppca_obj):

    if torch.is_tensor(x):
        x = x.detach().numpy()

    M_inv = ppca_obj[1]
    W = ppca_obj[2]

    z = (M_inv @ W.T @ x.T).T
    x_est = z @ W.T

    return x_est

class VAE(torch.nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, latent_dim, sigmas_init):
        super(VAE, self).__init__()

        self.log_sigmas = torch.log(sigmas_init)

        self.e1 = nn.Linear(input_dim, h1_dim)
        self.e2 = nn.Linear(h1_dim, h2_dim)
        self.e_mean = nn.Linear(h2_dim, latent_dim)
        self.e_log_var = nn.Linear(h2_dim, latent_dim)

        self.d1 = nn.Linear(latent_dim, h2_dim)
        self.d2 = nn.Linear(h2_dim, h1_dim)
        self.d_mean = nn.Linear(h1_dim, input_dim)

    def encoder(self, x):

        z_mean = self.e_mean(F.relu(self.e2(F.relu(self.e1(x)))))
        z_log_var = self.e_log_var(F.relu(self.e2(F.relu(self.e1(x)))))

        return z_mean, z_log_var

    def decoder(self, z):

        x_mean = self.d_mean(F.relu(self.d2(F.relu(self.d1(z)))))

        return x_mean

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)

        return sample

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_fit = self.decoder(z)

        return x_fit, mu, log_var, self.log_sigmas


class skipVAE(torch.nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, latent_dim, sigmas_init):
        super(skipVAE, self).__init__()

        self.log_sigmas = torch.log(sigmas_init)

        self.e1 = nn.Linear(input_dim, h1_dim)
        self.e2 = nn.Linear(h1_dim, h2_dim)
        self.e_mean = nn.Linear(h2_dim, latent_dim)
        self.e_log_var = nn.Linear(h2_dim, latent_dim)

        self.d1 = nn.Linear(latent_dim, h2_dim)
        self.d2 = nn.Linear(h2_dim+latent_dim, h1_dim)
        self.d_mean = nn.Linear(h1_dim+latent_dim, input_dim)

    def encoder(self, x):

        z_mean = self.e_mean(F.relu(self.e2(F.relu(self.e1(x)))))
        z_log_var = self.e_log_var(F.relu(self.e2(F.relu(self.e1(x)))))

        return z_mean, z_log_var

    def decoder(self, z):

        d1_out = F.relu(self.d1(z))
        d1_out_cat = torch.cat((d1_out, z), dim=1)
        d2_out = F.relu(self.d2(d1_out_cat))
        d2_out_cat = torch.cat((d2_out, z), dim=1)

        x_mean = self.d_mean(d2_out_cat)

        return x_mean

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)

        return sample

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_fit = self.decoder(z)

        return x_fit, mu, log_var, self.log_sigmas


def loss_function(x, x_fit, mu, log_var, log_sigmas, sig_df, sig_scale):

    sigmas = torch.exp(log_sigmas)

    x_sig = torch.div(x, sigmas)
    x_fit_sig = torch.div(x_fit, sigmas)

    rec_loss = 0.5 * F.mse_loss(x_sig, x_fit_sig, reduction='sum')

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return rec_loss + kl_loss

def likelihood(x, x_fit, sigmas):

    if torch.is_tensor(x):
        x = x.detach().numpy()

    if torch.is_tensor(x_fit):
        x_fit = x_fit.detach().numpy()

    if torch.is_tensor(sigmas):
        sigmas = sigmas.detach().numpy()

    x_sig = np.divide(x, sigmas)
    x_fit_sig = np.divide(x_fit, sigmas)

    ll = 0.5 * ((x_sig - x_fit_sig) ** 2).mean() * x_fit.shape[1]

    return ll

#----------------------------------------------------

N = 5000

input_dim = 10
latent_dim = 2

batch_size = 100
nepoch = 200

lr = 1e-3

sigma_true = 1

W = torch.zeros(input_dim, latent_dim)
W[0:5, 0] = 5
W[5:10, 1] = 5

W_t = W.t()

z_train = torch.randn(N, latent_dim)
x_train = torch.matmul(z_train, W_t) + (sigma_true) * torch.randn(N, input_dim)

z_test = torch.randn(N, latent_dim)
x_test = torch.matmul(z_test, W_t) + (sigma_true) * torch.randn(N, input_dim)

z_val = torch.randn(N, latent_dim)
x_val = torch.matmul(z_train, W_t) + (sigma_true) * torch.randn(N, input_dim)

dataloader_train = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=True)

# ppca

ppca_train = ppca(x_train, latent_dim)
ppca_test = ppca(x_test, latent_dim)

# set up sigmas prior
x_np = x_train.numpy()
sigmas_init = torch.from_numpy(np.array(1.0))

sig_df = 1
sig_scale = 1

vae_train = VAE(input_dim, 10, 10, latent_dim, sigmas_init)
optimizer_vae_train = optim.Adam(vae_train.parameters(), lr=lr, weight_decay=1.2e-6)

l = None

for epoch in range(nepoch):
    for batch_idx, data in enumerate(dataloader_train):
        optimizer_vae_train.zero_grad()
        x_fit, mu, log_var, log_sigmas = vae_train(data)

        loss = loss_function(data, x_fit, mu, log_var, log_sigmas, sig_df, sig_scale)
        loss.backward()
        optimizer_vae_train.step()

        l = loss.item()

    print(epoch, l)

x_np = x_test.numpy()
sigmas_init = torch.from_numpy(np.array(1.0))

vae_test = VAE(input_dim, 10, 10, latent_dim, sigmas_init)
optimizer_vae_test = optim.Adam(vae_test.parameters(), lr=lr, weight_decay=1.2e-6)

l = None

for epoch in range(nepoch):
    for batch_idx, data in enumerate(dataloader_test):
        optimizer_vae_test.zero_grad()
        x_fit, mu, log_var, log_sigmas = vae_test(data)

        loss = loss_function(data, x_fit, mu, log_var, log_sigmas, sig_df, sig_scale)
        loss.backward()
        optimizer_vae_test.step()

        l = loss.item()

    print(epoch, l)

x_np = x_train.numpy()
sigmas_init = torch.from_numpy(np.array(1.0))

skip_train = skipVAE(input_dim, 10, 10, latent_dim, sigmas_init)
optimizer_skip_train = optim.Adam(skip_train.parameters(), lr=lr, weight_decay=1.2e-6)

l = None

for epoch in range(nepoch):
    for batch_idx, data in enumerate(dataloader_train):
        optimizer_skip_train.zero_grad()
        x_fit, mu, log_var, log_sigmas = skip_train(data)

        loss = loss_function(data, x_fit, mu, log_var, log_sigmas, sig_df, sig_scale)
        loss.backward()
        optimizer_skip_train.step()

        l = loss.item()

    print(epoch, l)

x_np = x_test.numpy()
sigmas_init = torch.from_numpy(np.array(1.0))

skip_test = skipVAE(input_dim, 10, 10, latent_dim, sigmas_init)
optimizer_skip_test = optim.Adam(skip_test.parameters(), lr=lr, weight_decay=1.2e-6)

l = None

for epoch in range(nepoch):
    for batch_idx, data in enumerate(dataloader_test):
        optimizer_skip_test.zero_grad()
        x_fit, mu, log_var, log_sigmas = skip_test(data)

        loss = loss_function(data, x_fit, mu, log_var, log_sigmas, sig_df, sig_scale)
        loss.backward()
        optimizer_skip_test.step()

        l = loss.item()

    print(epoch, l)

#------------------------------------------

n_rep = 500
n_models = 3

ll_pp_data = np.zeros((n_models, n_models, n_rep))

for k in range(n_models):
    for l in range(n_models):
        for r in range(n_rep):

            if k == 0:
                z = ppca_train[0]
                z = z + np.random.multivariate_normal(mean=np.zeros(latent_dim), cov=ppca_train[3] * ppca_train[1], size = N)
                pp_data = z @ ppca_train[2].T
                pp_data = pp_data + np.sqrt(ppca_train[3]) * np.random.randn(N, input_dim)

            if k == 1:
                temp_mean, temp_log_var = vae_train.encoder(x_train)
                z = temp_mean + torch.sqrt(temp_log_var.exp()) * torch.randn(N, latent_dim)
                x_mean = vae_train.decoder(z)
                pp_data = x_mean + (vae_train.log_sigmas.exp()) * torch.randn(N, input_dim)

            if k == 2:
                temp_mean, temp_log_var = skip_train.encoder(x_train)
                z = temp_mean + torch.sqrt(temp_log_var.exp()) * torch.randn(N, latent_dim)
                x_mean = skip_train.decoder(z)
                pp_data = x_mean + (skip_train.log_sigmas.exp()) * torch.randn(N, input_dim)

            if l == 0:
                x_pred = ppca_est(pp_data, ppca_test)
                sigmas = np.sqrt(ppca_test[3])

            if l == 1:
                if not torch.is_tensor(pp_data):
                    pp_data = torch.from_numpy(pp_data)
                    pp_data = pp_data.float()
                z_mean, z_log_var = vae_test.encoder(pp_data)
                x_pred = vae_test.decoder(z_mean)
                sigmas = vae_test.log_sigmas.exp()

            if l == 2:
                if not torch.is_tensor(pp_data):
                    pp_data = torch.from_numpy(pp_data)
                    pp_data = pp_data.float()
                z_mean, z_log_var = skip_test.encoder(pp_data)
                x_pred = skip_test.decoder(z_mean)
                sigmas = skip_test.log_sigmas.exp()

            ll_pp_data[k, l, r] = likelihood(pp_data, x_pred, sigmas)


ll_partial = np.zeros(n_models)
partial_pc = np.zeros(n_models)

for k in range(n_models):
    if k == 0:
        x_val_pred = ppca_est(x_val, ppca_test)
        sigmas = ppca_test[3]


    if k == 1:
        z_mean, z_log_var = vae_test.encoder(x_val)
        x_val_pred = vae_test.decoder(z_mean)
        sigmas = vae_test.log_sigmas.exp()

    if k == 2:
        z_mean, z_log_var = skip_test.encoder(x_val)
        x_val_pred = skip_test.decoder(z_mean)
        sigmas = skip_test.log_sigmas.exp()

    ll_partial[k] = likelihood(x_val, x_val_pred, sigmas)

    partial_pc[k] = min((sum(ll_pp_data[k, k, :] > ll_partial[k])/n_rep, sum(ll_pp_data[k, k, :] < ll_partial[k])/n_rep))



x_val_ppca = ppca_est(x_val, ppca_test)
plt.scatter(x_val, x_val_ppca)

z_mean, z_log_var = vae_train.encoder(x_val)
x_val_vae = vae_train.decoder(z_mean)
x_val_vae = x_val_vae.detach().numpy()
plt.scatter(x_val, x_val_vae)

z_mean, z_log_var = skip_test.encoder(x_val)
x_val_skip = skip_test.decoder(z_mean)
x_val_skip = x_val_skip.detach().numpy()
plt.scatter(x_val, x_val_skip)

plt.scatter(x_val_ppca, x_val_skip)

save('out/linear_pp_data.npy', ll_pp_data)
save('out/linear_ll_partial.npy', ll_partial)
