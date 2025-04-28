import torch, pdb
import torch.nn as nn

import numpy as np

from src.Modules.Models.BuildMLP import BuildMLP
from src.Modules.Activations.SoftplusReLU import SoftplusReLU
from src.Modules.Utils.Gradient import Gradient
from torch.autograd import Variable
from src.Modules.Models.functionTerms import *

class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    a cell density dependent diffusion MLPs
    
    Inputs:
        name (string) : name of BINN
        x (nparray) : space grid
        t (nparray) : time grid
        
 
    '''
    
    def __init__(self, name, x, t, pde_weight = 1.0, D_max=1.0, G_max=1.0):
        
        super().__init__()
        
        # surface fitter
        self.surface_fitter = u_MLP(scale=1.0)
        
        # parameter extrema
        self.D_min = 0
        self.D_max = D_max
        self.G_min = 0
        self.G_max = G_max
        self.K     = 1.0# 1.7e3

        # input extrema
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.t_min = np.min(t)
        self.t_max = np.max(t)
        
        # pde functions
        if "DMLP" in name:
            self.diffusion = function_MLP(scale=1.0,fmin = self.D_min, fmax = self.D_max)
        elif "Dconst" in name:
            self.diffusion = param_const(scale=1.0,fmin = self.D_min, fmax = self.D_max)
        else:
            self.diffusion = NoFunction()
        
        if "GMLP" in name:
            self.growth = function_MLP(scale=1.0,  fmin = self.G_min, fmax = self.G_max)
        elif "Gconst" in name:
            self.growth = G_const(fmin = self.G_min, fmax = self.G_max, scale=1.0)
        else:
            self.growth = NoFunction()
        
                                 
        # loss weights
        self.surface_weight = 1.0
        self.pde_weight = pde_weight
        self.IC_weight = 10.0
        self.D_weight = 1e10 / self.D_max
        self.G_weight = 1e10 / self.G_max
        self.dDdu_weight = self.D_weight * self.K
        self.dGdu_weight = self.G_weight * self.K
        
        # number of samples for pde loss
        self.num_samples = 10000
        
        # model name
        self.name = name
    
    def forward(self, inputs):
        
        # cache input batch for pde loss
        self.inputs = inputs
        
        return self.surface_fitter(self.inputs)
    
    def gls_loss(self, pred, true):
        residual = (pred - true)**2
        
        '''#initial agent interior
        IC_interior = torch.logical_and(self.inputs[:, 0][:, None] >= .4*self.x_max,
                                        self.inputs[:, 0][:, None] <= .6*self.x_max)
        #initial agent exterior
        IC_exterior = torch.logical_not(IC_interior)
        
        #time zero
        t0 = self.inputs[:, 1][:, None] == self.t_min

        IC_interior_t0 = torch.logical_and(IC_interior,t0)
        IC_exterior_t0 = torch.logical_and(IC_exterior,t0)
        
        #Add residual for user-specified initial condition
        residual += self.IC_weight*torch.where(IC_interior_t0, torch.abs(pred - 0.75), torch.zeros_like(pred))
        residual += self.IC_weight*torch.where(IC_exterior_t0, torch.abs(pred - 0.0), torch.zeros_like(pred))'''
        
        # add weight to initial condition
        residual *= torch.where(self.inputs[:, 1][:, None] == self.t_min, 
                                self.IC_weight*torch.ones_like(pred), 
                                torch.ones_like(pred))
        
        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, return_mean=True):
        
        # unpack inputs
        x = inputs[:, 0][:,None]
        t = inputs[:, 1][:,None]

        # partial derivative computations 
        u = outputs.clone()
        d1 = Gradient(u, inputs, order=1)
        ux = d1[:, 0][:, None]
        ut = d1[:, 1][:, None]
        
        D = self.diffusion(u)
        G = self.growth(u)
        
        # Reaction-diffusion Equation
        LHS = ut
        RHS = Gradient(D*ux, inputs)[:, 0][:,None] + G*u
        pde_loss = (LHS - RHS)**2

        # constraints on learned parameters
        self.D_loss = 0
        self.G_loss = 0
        
        self.bdy_loss = 0
        
        self.D_loss += self.D_weight*torch.where(
            D < self.D_min, (D-self.D_min)**2, torch.zeros_like(D))
        self.D_loss += self.D_weight*torch.where(
            D > self.D_max, (D-self.D_max)**2, torch.zeros_like(D))
        self.G_loss += self.G_weight*torch.where(
            G < self.G_min, (G-self.G_min)**2, torch.zeros_like(G))
        self.G_loss += self.G_weight*torch.where(
            G > self.G_max, (G-self.G_max)**2, torch.zeros_like(G))
        
        '''dGdu =    Gradient(G, u,    order=1)
        self.G_loss += self.dGdu_weight*torch.where(
                dGdu > 0.0, dGdu**2, torch.zeros_like(dGdu))'''
        
        if return_mean:
            return torch.mean(pde_loss + self.D_loss + self.G_loss + self.bdy_loss)
        else:
            return pde_loss + self.D_loss + self.G_loss + self.bdy_loss
    
    def loss(self, pred, true):
        
        self.gls_loss_val = 0
        self.pde_loss_val = 0
        
        # load cached inputs from forward pass
        inputs = self.inputs
        
        # randomly sample from input domain
        x = torch.rand(self.num_samples, 1, requires_grad=True) 
        
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)
        
        outputs_rand = self.surface_fitter(inputs_rand)
        
        # compute surface loss
        self.gls_loss_val = self.surface_weight*self.gls_loss(pred, true)
        
        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand)
        
        return self.gls_loss_val + self.pde_loss_val
    
    
