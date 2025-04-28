import torch.nn as nn
import torch
from .utils import *

class LevelTrend(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    """

    def __init__(self, latent_dim = 3):
        super(LevelTrend, self).__init__()

        self.latent_dim = latent_dim
        self.dampening = nn.Parameter(torch.zeros(latent_dim))
        self.trend_mean = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, level_prev, trend_prev):

        damp = sigmoid(self.dampening, min = 0.0, max = 1.0)
        trend_mean = self.trend_mean

        return (level_prev + trend_prev, trend_mean + damp * (trend_prev - trend_mean ) )

    def forward_cov(self, level_scale, trend_scale, covariance, level_noise_scale, trend_noise_scale):

        damp = sigmoid(self.dampening, min = 0.0, max = 1.0)

        level_scale_next = torch.sqrt( level_scale**2 + trend_scale**2 + 2*covariance + level_noise_scale**2 )
        trend_scale_next = torch.sqrt( (damp * trend_scale)**2 + trend_noise_scale**2 )
        covariance_next = damp * covariance + damp * ( trend_scale**2 )

        return (level_scale_next, trend_scale_next, covariance_next)