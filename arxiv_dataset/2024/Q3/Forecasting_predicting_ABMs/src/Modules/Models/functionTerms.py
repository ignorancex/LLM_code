import torch, pdb
import torch.nn as nn

from src.Modules.Models.BuildMLP import BuildMLP
from src.Modules.Activations.SoftplusReLU import SoftplusReLU
from src.Modules.Utils.Gradient import Gradient
from torch.autograd import Variable

class u_MLP(nn.Module):
   
    '''
    Construct MLP surrogate model for the solution of the governing PDE. 
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted cell densities non-negative.
    
    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity
    
    Args:
        inputs (torch tensor): x and t pairs with shape (N, 2)
        
    Returns:
        outputs (torch tensor): predicted u values with shape (N, 1)
    '''
    
    def __init__(self, scale=1.7e3, output_features=1):
        
        super().__init__()
        self.scale = scale
        self.mlp = BuildMLP(
            input_features=2, 
            layers=[128, 128, 128, output_features],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
    
    def forward(self, inputs):
        
        outputs = self.scale * self.mlp(inputs)
        
        return outputs

    
class G_const(nn.Module):
    
    '''
    Construct logistic growth function.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        G (torch tensor): predicted growth values with shape (N, 1)
    '''
    
    def __init__(self, fmin, fmax, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = fmin
        self.max = fmax
        self.G = nn.Parameter(torch.rand(1))
    
    def forward(self, u, t=None):
        
        if t is None:
            G = self.G*(torch.ones_like(u) - u/self.scale)
        else:
            G = self.G*(torch.ones_like(u) - u/self.scale)
        G = self.max * G
        
        return G    
    

class NoFunction(nn.Module):
    
    '''
    Trivial function.
    
    u (torch tensor): predicted u values with shape (N, 1)
    t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        F (torch tensor): zeros with shape (N, 1)
    '''
    
    
    def __init__(self, fmin=0, fmax=1.0, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = fmin
        self.max = fmax
        
    def forward(self, u, t=None):
        
        if t is None:
            F = torch.zeros_like(u)
        else:
            F = torch.zeros_like(torch.zeros_like(torch.cat([u,t], dim=1)))
        return F
    
class NoDelay(nn.Module):
    
    '''
    Trivial delay function.
    
    Args:
        t (torch tensor): time values with shape (N, 1)
        
    Returns:
        T (torch tensor): ones with shape (N, 1)
    '''
    
    
    def __init__(self):
        
        super().__init__()
        
    def forward(self, t):
        
        T = torch.ones_like(t)
        
        return T

class T_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown time delay function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output is 
    linearly sigmoid-activated to constrain outputs to between 0 and 1.
    
    Args:
        t (torch tensor): time values with shape (N, 1)
        
    Returns:
        T (torch tensor): predicted delay values with shape (N, 1)
    '''
    
    
    def __init__(self):
        
        super().__init__()
        self.mlp = BuildMLP(
            input_features=1, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=nn.Sigmoid())
        
    def forward(self, t):
        
        T = self.mlp(t) 
        
        return T

class function_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted diffusivities non-negative.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        D (torch tensor): predicted diffusivities with shape (N, 1)
    '''
    
    
    def __init__(self, fmin, fmax, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = fmin
        self.max = fmax
        self.mlp = BuildMLP(
            input_features=input_features, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
        
    def forward(self, u, t=None):
        
        if t is None:
            MLP = self.mlp(u/self.scale)
        else:
            MLP = self.mlp(torch.cat([u/self.scale, t], dim=1))    
        MLP = self.max * MLP
        
        return MLP
    
class param_const(nn.Module):
    
    '''
    Construct unknown trainable constant. 
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        D (torch tensor): predicted diffusivities with shape (N, 1)
    '''
    
    def __init__(self, fmin, fmax, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = fmin
        self.max = fmax
        self.param = nn.Parameter(torch.rand(1))
        
    def forward(self, u, t=None):
        
        if t is None:
            f = self.param * torch.ones_like(u)
        else:
            f = self.param * torch.ones_like(torch.cat([u, t], dim=1))    
        f = self.max * f
        
        return f       