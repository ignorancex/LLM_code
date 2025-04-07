import autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
import autograd.numpy as np


class VAEncoder(nn.Module):
  def __init__(self, input_dim, latent_dim, layer_dim):
    super(VAEncoder, self).__init__()
    self.latent_dim = latent_dim

    # Encoder Net
    self.encoder_net = nn.Sequential(nn.Linear(input_dim,layer_dim[0]),
                                nn.ELU(inplace=True),
                                nn.Linear(layer_dim[0],layer_dim[1]),
                                nn.ELU(inplace=True)
                                )
    # parameterize latent distribution
    self.mu_net        = nn.Linear(layer_dim[1],latent_dim)
    self.log_var_net   = nn.Linear(layer_dim[1],latent_dim)

  def encoder(self, x):  
    x              = self.encoder_net(x)
    self.mu        = self.mu_net(x)
    self.log_var   = self.log_var_net(x)
    self.var       = torch.exp(self.log_var) #torch.exp(0.5 * self.log_var)   

    # Reparametrization trick      
    epsilon = torch.randn(x.size(0), self.latent_dim)
    z_sample = self.mu + self.var * epsilon 

    return z_sample

  def forward(self, x):

    self.latent_r = self.encoder(x)
      
    # Compute KL loss
   # self.kl_loss = kl_divergence(self.mu, self.log_var, dim=self.latent_dim) 
    return self.latent_r

  # KL div varriant used in SciNet: https://arxiv.org/abs/1807.10300
  def kl_divergence(self, means, log_sigma, dim, target_sigma=0.1):
    """
    Computes Kullback–Leibler divergence for arrays of mean and log(sigma)

    1/2(1/sigma_target^2 * mu^2 + e^(2* log(sigma))/sigma_target^2 + 2*log(sigma_target) - latent_dim)
    """
    target_sigma = torch.Tensor([target_sigma])
    kl_loss = 1 / 2. * torch.mean(torch.mean(1 / target_sigma**2 * means**2 +
        torch.exp(2 * log_sigma) / target_sigma**2 - 2 * log_sigma + 2 * torch.log(target_sigma), dim=1) - dim)

    return kl_loss



class MLP(nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, activation_func):
    super(MLP, self).__init__()

    self.net = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                             activation_func,
                             nn.Linear(hidden_dim,hidden_dim),
                             activation_func,
                             nn.Linear(hidden_dim, output_dim, bias=None)
                             )
    
    self.net.apply(self.init_weights)

  def forward(self, x):
    x = self.net(x)
    return x

  def init_weights(self,layer):
    if type(layer) == nn.Linear:
      torch.nn.init.orthogonal_(layer.weight)

  def time_derivative(self,x,qp, t=None):
    x = self.net(x)
    return x



class HNN(torch.nn.Module):
    """
    :differentiable_model: MLP(input_dim =2,hidden_dim = 200,output_dim = 1), H_theta
    :input_dim: number of canonical coordinates (q,p) given as input, the latent variables output from the encoder are also passed to the differentiable_model
    """
    
    def __init__(self, input_dim, differentiable_model):    
        super(HNN, self).__init__()
        
        self.input_dim = input_dim
        self.differentiable_model = differentiable_model # MLP model
        
    def forward(self, x):
        # traditional forward pass

        H_theta = self.differentiable_model(x)
        
        return H_theta

    def time_derivative(self, x,qp, t=None):

        H_theta = self.forward(x) # MLP parameterization of the hamiltonian returns an energy like value H*
        H_partials = torch.autograd.grad(H_theta.sum(), qp, create_graph=True)[0] # returns partials w.r.t (q & p) only, latent var partials are not returned

        #Map partials: dH/dp => dq/dt, -dH/dq => dp/dt
        dx = self.canoncial_map(H_partials,n = self.input_dim)
        
        return dx

    def canoncial_map(self,H_partials,n):
      """
      Maps:
       dH/dp  =>  dq/dt
      -dH/dq  =>  dp/dt
      """
      M = torch.eye(2)
      M = torch.cat([M[n//2:], -M[:n//2]])
      return H_partials @ M.t()


class ConSciNet(nn.Module):

  def __init__(self, Encoder, NAFunc,trial):
    super(ConSciNet, self).__init__()
    self.trial = trial

    self.encoder = Encoder
    self.nafunc =  NAFunc
    
  def forward(self, x):

    aux_vars = x[:,-2:] 
    aux_vars.requires_grad = True
    x = x[:,0:-2] 
    z = self.encoder(x)
    nafunc_in = torch.cat([aux_vars,z], dim = 1) # f_decoder(q,p,z)
    time_d_out = self.nafunc.time_derivative(nafunc_in,aux_vars)

    return time_d_out
