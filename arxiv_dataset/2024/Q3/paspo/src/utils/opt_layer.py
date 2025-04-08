import torch

from qpth.qp import QPFunction
from torch.autograd import Variable

import numpy as np

class OptLayer(torch.nn.Module):
    def __init__(self, n_dimensions, A, b, device, normalize=True):
        super(OptLayer, self).__init__()
        self.normalize = normalize #original paper uses normalization

        #attention Gx <= h are the inequality constraints (which in our naming are the constraints of the form Ax <= b)
        # Ax == b are the equality constraints 
        self.Q = Variable(torch.eye(n_dimensions, dtype=torch.float64).to(device))
        
        
        # Add constraints for 0 <= x <= 1
        A = np.vstack([A, -np.eye(n_dimensions), np.eye(n_dimensions)])
        b = np.hstack([b, np.zeros(n_dimensions), np.ones(n_dimensions)])

        # Gx <= h
        self.G = Variable(torch.tensor(A, dtype=torch.float64).to(device))
        
        self.h = Variable(torch.tensor(b, dtype=torch.float64).to(device))
        # Ax == b
        self.A = Variable(torch.ones((1, n_dimensions), dtype=torch.float64).to(device))
        self.b = Variable(torch.tensor([1], dtype=torch.float64).to(device))

    def forward(self, x):
        batch_size = x.shape[0]

        Q = self.Q.repeat(batch_size, 1, 1)
        G = self.G.repeat(batch_size, 1, 1)
        h = self.h.repeat(batch_size, 1)
        A = self.A.repeat(batch_size, 1, 1)
        b = self.b.repeat(batch_size, 1)
        
        #notImprovedLim=1, maxIter=20,  eps=1e-6
        return QPFunction(verbose=-1, notImprovedLim=1, maxIter=20,  eps=1e-6)(Q, x.double(), G, h, A, b).float()
    
    def compute_cost(self, action_unsafe):
        batch_size = action_unsafe.shape[0]
        
        if self.normalize:
            cost_eq = ((self.A / self.b.abs().unsqueeze(-1)).repeat(batch_size, 1,1) @ action_unsafe.unsqueeze(-1).double()).sum(dim=-1) - 1

            cost_ineq = torch.max(((self.G / self.h.unsqueeze(-1).abs()).repeat(batch_size, 1,1) @ action_unsafe.unsqueeze(-1).double()).sum(dim=-1) - 1, torch.tensor(0, dtype=action_unsafe.dtype, device=action_unsafe.device))
        else:
            cost_eq = (self.A.repeat(batch_size, 1,1) @ action_unsafe.unsqueeze(-1).double()).sum(dim=-1) - self.b
            cost_ineq = torch.max((self.G.repeat(batch_size, 1,1) @ action_unsafe.unsqueeze(-1).double()).sum(dim=-1) - self.h, torch.tensor(0, dtype=action_unsafe.dtype, device=action_unsafe.device))
        
        total_cost = cost_eq.sum(dim=-1) + cost_ineq.sum(dim=-1)
        return total_cost
    
    
if __name__ == "__main__":
    n_dimensions = 3
    from utils.polytope_loader import load_polytope

    A,b = load_polytope(n_dim=n_dimensions,
                        storage_method="seed", 
                        generation_method="points", 
                        polytope_generation_data={"seed": 1, "n_points": 30})
    
    action = torch.tensor([0.4976, 0.4677, 0.0347], dtype=torch.float64, device="cpu")
    #action = torch.tensor([0.1691, 0.1190, 0.6320], dtype=torch.float64, device="cpu")

    opt_layer = OptLayer(n_dimensions, A, b, "cpu")


    x = opt_layer(action.unsqueeze(0))

    a = 3