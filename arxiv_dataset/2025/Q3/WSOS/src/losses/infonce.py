import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, batch1, batch2):
        # Ensure the batches have the same number of samples
        assert batch1.shape[0] == batch2.shape[0], "The two batches must have the same number of samples"

        sim_matrix = torch.matmul(batch1, batch2.T) / self.temperature
        #https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        # -log p_i + logsumexp([p_1,...,p_i,..,p_n]) <-- InfoNCEloss
        return (-sim_matrix.diagonal() + torch.logsumexp(sim_matrix, dim=1)).mean()