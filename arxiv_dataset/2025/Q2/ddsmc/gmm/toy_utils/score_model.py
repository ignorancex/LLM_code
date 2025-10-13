import torch

class ScoreModelDDSMC(torch.nn.Module):
    def __init__(self, score_net, Vt=None):
        super().__init__()
        self.model = score_net
        self.Vt = Vt
    
    def score(self, x, sigma):
        if self.Vt is not None:
            x_regular_basis = (self.Vt.T @ x.T).T
            score = (self.Vt @ self.model(x_regular_basis, sigma).T).T
        else:
            score = self.model(x, sigma)
        return score
