import torch
import torch.nn as nn

class Emitter(nn.Module):
    #Parameterizes `p(x_t | z_t)`

    def __init__(self, input_dim, param, model_type = 'radial_basis', latent_dim = 3):
        super(Emitter, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if model_type == 'linear':
            self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim))
        elif model_type == 'radial_basis':
            num_basis = param['num_basis']
            tau = param['tau']
            mu = torch.tensor(range(num_basis))/num_basis
            self.mu = nn.Parameter(mu.unsqueeze(0).repeat(latent_dim,1))
            self.w = nn.Parameter(torch.empty(latent_dim,num_basis))
            nn.init.uniform_(self.w)
            self.l = torch.ones(latent_dim,num_basis)*tau
            self.bias = nn.Parameter(torch.ones(input_dim))
            self.decoder = self.radial_basis
    
    def radial_basis(self, x):
        ages = torch.tensor(range(self.input_dim)).reshape(self.input_dim,1,1)/self.input_dim
        p = torch.exp(-torch.pow((ages-self.mu)*self.l,2))
        return torch.sum(p*self.w,2) @ x + self.bias

    def forward(self, x):
        return torch.exp(self.decoder(x))
