import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
    
    def forward_gen(self, disc_outputs):
        L_perc_disc = sum(
            F.l1_loss(fake_x, real_x)
            for fake_x, real_x in zip(disc_outputs['fake_features'],
                                      disc_outputs['real_features'])
        )
        
        L_adv_G = -disc_outputs['fake_score_G'].mean()
        
        return {
            'L_perc_disc': L_perc_disc,
            'L_adv_G': L_adv_G
        }
        
    def forward_disc(self, disc_outputs):
        L_adv_D = (
            torch.relu(1. - disc_outputs['real_score']).mean() +
            torch.relu(1. + disc_outputs['fake_score_D']).mean()
        )
        return L_adv_D