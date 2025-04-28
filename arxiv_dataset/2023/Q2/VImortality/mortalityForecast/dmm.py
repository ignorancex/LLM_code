import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from .utils import *

class DMM(nn.Module):

    def __init__(self, num_years, input_dim, emitter, latent_transition):
        super(DMM, self).__init__()
        
        self.num_years = num_years
        self.input_dim = input_dim

        self.emitter = emitter
        self.latent_transition = latent_transition

        self.latent_dim = self.emitter.latent_dim

        self.scale_transform = nn.Softplus()
        self.scale_transform_inv = softplus_inv

        #Model parameters
        self.level_0 = nn.Parameter(torch.zeros(self.latent_dim))
        self.trend_0 = nn.Parameter(torch.zeros(self.latent_dim))
        self.level_scale = nn.Parameter(torch.ones(self.latent_dim))
        self.trend_scale = nn.Parameter(torch.ones(self.latent_dim))

        #Guide parameters
        self.guide_level_mean = nn.Parameter(torch.zeros(self.latent_dim,num_years))
        self.guide_level_scale = nn.Parameter(torch.ones(self.latent_dim,num_years)*0.001)
        self.guide_level_damp = nn.Parameter(torch.zeros(self.latent_dim,num_years-1))
        self.guide_trend_mean = nn.Parameter(torch.zeros(self.latent_dim,num_years))
        self.guide_trend_scale = nn.Parameter(torch.ones(self.latent_dim,num_years)*0.001)
        self.guide_trend_damp = nn.Parameter(torch.zeros(self.latent_dim,num_years-1))
        self.guide_corr = nn.Parameter(torch.zeros(self.latent_dim,num_years))

    def model(self, data):

        deaths, exposure = data
        
        pyro.module("dmm", self)

        level_prev = self.level_0
        trend_prev = self.trend_0

        for t in range(self.num_years):

            (level_loc, trend_loc) = self.latent_transition(level_prev, trend_prev)
                
            # scale is standard deviation    
            level = pyro.sample("level_%d" % t, dist.Normal(level_loc, self.scale_transform(self.level_scale) ).to_event(1))
            trend = pyro.sample("trend_%d" % t, dist.Normal(trend_loc, self.scale_transform(self.trend_scale) ).to_event(1))

            # compute the intensities that parameterize the Poisson likelihood
            intensity_t = self.emitter(level)
            
            for i in range( len(deaths[t, :]) ):
                if exposure[t, i] > 0 and intensity_t[i] > 0 :
                    pyro.sample("obs_%d_%d" % (t, i+1),
                            dist.Poisson(intensity_t[i] * exposure[t, i], validate_args = False),
                            obs=deaths[t, i])

            level_prev = level
            trend_prev = trend

    def guide(self, data):

        deaths, exposure = data

        level_vec =  torch.zeros(self.latent_dim, self.num_years)
        trend_vec = torch.zeros(self.latent_dim, self.num_years)

        pyro.module("dmm", self)

        level_mean = self.guide_level_mean[:,0]
        level_scale = self.scale_transform(self.guide_level_scale[:,0])

        level = pyro.sample("level_0", 
                            dist.Normal(level_mean, level_scale)
                            .to_event(1))

        corr = sigmoid(self.guide_corr[:,0], min = -1.0, max = 1.0)

        trend_mean = self.guide_trend_mean[:,0] + corr * level
        trend_scale = self.scale_transform(self.guide_trend_scale[:,0])

        trend = pyro.sample("trend_0", 
                            dist.Normal(trend_mean, trend_scale )
                            .to_event(1))

        level_vec[:,0] = level
        trend_vec[:,0] = trend

        for t in range(1, self.num_years):

            corr = sigmoid(self.guide_corr[:,t], min = -1.0, max = 1.0)
            
            level_damp = sigmoid(self.guide_level_damp[:,t-1], min = 0.0, max = 1.0)
            trend_damp = sigmoid(self.guide_trend_damp[:,t-1], min = 0.0, max = 1.0)

            level_mean = self.guide_level_mean[:,t] + level_damp * level
            level_scale = self.scale_transform(self.guide_level_scale[:,t])

            level = pyro.sample("level_%d" % (t), 
                              dist.Normal(level_mean, level_scale )
                              .to_event(1))

            trend_mean = self.guide_trend_mean[:,t] + trend_damp * trend + corr * level
            trend_scale = self.scale_transform(self.guide_trend_scale[:,t])

            trend = pyro.sample("trend_%d" % (t), 
                              dist.Normal(trend_mean, trend_scale )
                              .to_event(1))

            level_vec[:,t] = level
            trend_vec[:,t] = trend

        return (level_vec, trend_vec)
