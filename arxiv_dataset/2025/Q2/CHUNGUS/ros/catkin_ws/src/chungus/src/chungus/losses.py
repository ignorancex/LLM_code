import torch
import torch.nn as nn
import torch.nn.functional as F


class LRIZZ(nn.Module):
    def __init__(self, l=0.5, reconstruct_weight=0.1):
        """ L-RIZZ loss with reconstruction term
        
        :param l: L-RIZZ l hyperparameter
        :param reconstruct_weight: weight for reconstruction term
        """
        super().__init__()
        self._l = l
        self._reconstruct_weight = reconstruct_weight
    
    def forward(self, predictionsA, predictionsB, targets):
        """ Compute the loss """

        # compute lrizz
        diff = predictionsB['prediction'] - predictionsA['prediction']
        loss_ineq = torch.square(F.relu(self._l - targets*diff)) * (targets != 0)
        loss_eq = torch.square(diff) * (targets == 0)
        total_loss = (loss_ineq + loss_eq).mean()

        # compute reconstruction loss if applicable
        if 'reconstruction' in predictionsA and 'reconstruction' in predictionsB:
            total_loss = total_loss + self._reconstruct_weight * (predictionsA['reconstruction'].mean() + predictionsB['reconstruction'].mean())

        return total_loss
