from .emitter import Emitter
from .level_trend import LevelTrend
from .dmm import DMM
from .utils import *

from os.path import exists

import torch
import numpy as np

import time

from scipy.stats import poisson
from scipy.special import logsumexp

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceMeanField_ELBO, config_enumerate
from pyro.optim import ClippedAdam, Adam

class Mortality:

    def __init__(self, param):

        if param['model'] == 'linear':
            self.emitter = Emitter(
                input_dim = param['max_age'] + 1, 
                param = None,
                model_type = 'linear', 
                latent_dim = param['latent_dim']
                )
        elif param['model'] == 'radial_basis':
            p ={
                "num_basis" : param['num_basis'], 
                "tau" : param['tau'],
                }
            self.emitter = Emitter(
                input_dim = param['max_age'] + 1, 
                param = p,
                model_type = 'radial_basis', 
                latent_dim = param['latent_dim']
                )

        self.latent_transition = LevelTrend(latent_dim = param['latent_dim'])

        self.latent_dim = param['latent_dim']
        self.num_train_years = param['last_year_train'] - param['first_year_train'] + 1
        self.num_test_years = param['last_year_test'] - param['first_year_test'] + 1

    def fit(self, exposure, deaths, 
            num_steps = 1000, 
            log_freq = 100, 
            checkpoint_freq = 100, 
            lr = 0.01,
            lr_decay = 0.1,
            load_file = None, 
            save_file = None):

        self.num_ages = exposure.shape[1]
        self.num_years = exposure.shape[0]

        exposure = torch.from_numpy(exposure)
        deaths = torch.from_numpy(deaths)
            
        self.data = (deaths, exposure)

        pyro.clear_param_store()

        self.dmm = DMM(num_years = self.num_years, input_dim = self.num_ages, emitter = self.emitter, latent_transition  = self.latent_transition)

        adam_params = {
        'betas' : (0.95, 0.999),
        "lr": lr,
        "lrd": lr_decay,
        "clip_norm": 10,
        "weight_decay": 0.0
        }

        self.optimizer = ClippedAdam(adam_params)

        if load_file is not None:
            self.load_checkpoint(load_file)

        svi = SVI(self.dmm.model, self.dmm.guide, self.optimizer, loss=TraceMeanField_ELBO())

        vi_times = [time.time()]

        self.train_loss = []
        loss = 0.0

        for step in range(num_steps):

            loss += svi.step(self.data)

            # if (step + 1) % decay_step == 0: # decay learning rate
            #     self.scheduler.step()

            if (step + 1) % log_freq == 0: # report training diagnostics
                # print(self.dmm.emitter.w)
                # print(self.dmm.emitter.mu)
                # print(self.dmm.emitter.l)

                vi_times.append(time.time())
                self.train_loss.append(loss/checkpoint_freq)
                epoch_time = vi_times[-1] - vi_times[-2]
                
                print("[step {:04}]  {:.4e} \t(dt = {:.4f} sec \t t =  {:.1f}) min".format(step+1, loss/log_freq, epoch_time, (vi_times[-1] - vi_times[0])/60))
                
                loss = 0.0

            if (step + 1) % log_freq == 0 and save_file is not None:
                self.save_checkpoint(save_file)

        return self

    def load(self, file_name, exposure, deaths):

        self.num_ages = exposure.shape[1]
        self.num_years = exposure.shape[0]

        exposure = torch.from_numpy(exposure)
        deaths = torch.from_numpy(deaths)

        self.data = (deaths, exposure)

        self.dmm = DMM(num_years = self.num_years, input_dim = self.num_ages, emitter = self.emitter, latent_transition  = self.latent_transition)

        adam_params = {
        }

        self.optimizer = ClippedAdam(adam_params)
        
        self.load_checkpoint(file_name)

        return self

    def load_checkpoint(self, file):

        load_model = file + ".model"
        load_opt = file + ".opt"

        assert exists(load_opt) and exists(load_model), \
            "--load-model and/or --load-opt misspecified"
        self.dmm.load_state_dict(torch.load(load_model))
        self.optimizer.load(load_opt)

    def save_checkpoint(self, file):

        save_model = file + ".model"
        save_opt = file + ".opt"

        torch.save(self.dmm.state_dict(), save_model)
        self.optimizer.save(save_opt)

    @property
    def level_loc(self):

        level_mean = self.dmm.guide_level_mean

        out = torch.zeros((self.dmm.latent_dim, self.dmm.num_years))
        out[:,0] = level_mean[:,0]

        for i in range(self.dmm.num_years-1):
            level_damp = sigmoid(self.dmm.guide_level_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = level_mean[:,i+1] + level_damp * out[:,i]

        return out.detach()

    @property
    def level_scale(self):

        level_scale = self.dmm.scale_transform(self.dmm.guide_level_scale)

        out = torch.zeros((self.dmm.latent_dim, self.dmm.num_years))
        out[:,0] = level_scale[:,0]

        for i in range(self.dmm.num_years-1):
            level_damp = sigmoid(self.dmm.guide_level_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = torch.sqrt(level_scale[:,i+1]**2 + (level_damp * out[:,i])**2)

        return out.detach()

    @property
    def level_trend_cov(self):
        
        corr = sigmoid(self.dmm.guide_corr[:,0], min = -1.0, max = 1.0)
        level_scale = self.dmm.scale_transform(self.dmm.guide_level_scale[:,0])

        out = torch.zeros((self.dmm.latent_dim, self.dmm.num_years))
        out[:,0] = corr * (level_scale**2)

        for i in range(self.dmm.num_years-1):
            corr = sigmoid(self.dmm.guide_corr[:,i+1], min = -1.0, max = 1.0)

            trend_damp = sigmoid(self.dmm.guide_trend_damp[:,i], min = 0.0, max = 1.0)
            level_damp = sigmoid(self.dmm.guide_level_damp[:,i], min = 0.0, max = 1.0)
            
            level_scale = self.dmm.scale_transform(self.dmm.guide_level_scale[:,i+1])

            out[:,i+1] = level_damp*trend_damp*out[:,i] + corr * level_damp * level_scale**2

        return out.detach()

    @property
    def trend_loc(self):

        trend_mean = self.dmm.guide_trend_mean 
        corr = sigmoid(self.dmm.guide_corr[:,0], min = -1.0, max = 1.0)

        level_mean = self.level_loc

        out = torch.zeros((self.dmm.latent_dim,self.dmm.num_years))
        out[:,0] = trend_mean[:,0] + corr * level_mean[:,0]

        for i in range(self.dmm.num_years-1):
            corr = sigmoid(self.dmm.guide_corr[:,i+1], min = -1.0, max = 1.0)
            trend_damp = sigmoid(self.dmm.guide_trend_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = trend_mean[:,i+1] + trend_damp*out[:,i] + corr * level_mean[:,i+1]

        return out.detach()

    @property
    def trend_scale(self):

        guide_trend_scale = self.dmm.scale_transform(self.dmm.guide_trend_scale)
        guide_corr = sigmoid(self.dmm.guide_corr[:,0], min = -1.0, max = 1.0)
        level_scale = self.level_scale
        level_trend_cov = self.level_trend_cov

        out = torch.zeros( (self.dmm.latent_dim, self.dmm.num_years) )

        out[:,0] = torch.sqrt( guide_trend_scale[:,0]**2 + ( guide_corr * level_scale[:,0] )**2 )

        for i in range(self.dmm.num_years-1):
            corr = sigmoid(self.dmm.guide_corr[:,i+1], min = -1.0, max = 1.0)
            trend_damp = sigmoid(self.dmm.guide_trend_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = torch.sqrt( (guide_trend_scale[:,i+1])**2 + (trend_damp * out[:,i])**2 + (corr*level_scale[:,i+1])**2  + 2*corr*level_trend_cov[:,i+1])

        return out.detach()

    @property
    def dampening(self):
        return sigmoid(self.dmm.latent_transition.dampening, min = 0.0, max = 1.0)

    @property
    def correlation(self):
        return sigmoid(self.dmm.guide_corr, min = -1.0, max = 1.0)

    @property
    def guide_sample(self):
        (level, trend) = self.dmm.guide(self.data)
        return ( level.detach(), trend.detach() )
    
    def mortality_curve(self, time):
        return self.dmm.emitter.forward(self.level_loc[:,time]).detach()

    @property
    def factor_loadings(self):

        epsilon = 0.01

        #Bias term
        out = [torch.log(self.dmm.emitter.forward(torch.zeros(self.dmm.latent_dim)).detach())]

        for i in range(self.dmm.latent_dim):
            z = torch.zeros(self.dmm.latent_dim)
            z[i] = epsilon

            out.append( (torch.log( self.dmm.emitter.forward(z).detach() ) - torch.log( self.dmm.emitter.forward( torch.zeros(self.dmm.latent_dim) )).detach())/epsilon )

        return out

    def smoothed_latent_mortality(self, age, time):
        return self.dmm.emitter.forward(self.level_loc[:,time]).detach()[age]

    def mortality(self, age, mc_samples = 1000):
        exposure = self.data[1]

        mean_mortality = np.zeros(self.num_years)
        std_mortality = np.zeros(self.num_years)

        level_loc = self.level_loc
        level_scale = self.level_scale

        for t in range(self.num_years):
            mortality = np.zeros(mc_samples)
            for i in range(mc_samples):
                x = torch.distributions.Normal(level_loc[:,t] , level_scale[:,t]).sample()
                intensity = self.dmm.emitter.forward(x)[age]
                mortality[i] = torch.distributions.Poisson(intensity * exposure[t,age]).sample() / exposure[t,age]
            mean_mortality[t] = np.mean(mortality)
            std_mortality[t] = np.std(mortality)
        
        return (mean_mortality, std_mortality)

    def forecast_mortality_rate(self, age, forecast_length, quantiles = [0.25, 0.75,], mc_samples = 1000):

        exposure = self.data[1]

        (level_loc_forecast, level_scale_forecast) = self.forecast_factors(forecast_length)

        mean_mortality = np.empty(forecast_length)
        std_mortality = np.empty(forecast_length)
        quantiles_mortality = np.empty([forecast_length,len(quantiles)])

        for t in range(forecast_length):
            mortality = np.empty(mc_samples)
            for i in range(mc_samples):
                factors = torch.distributions.Normal(level_loc_forecast[:,t] , level_scale_forecast[:,t]).sample()
                intensity = self.dmm.emitter.forward(factors)[age]
                mortality[i] = torch.distributions.Poisson(intensity * exposure[t,age]).sample() / exposure[t,age]
            mean_mortality[t] = np.mean(mortality)
            std_mortality[t] = np.std(mortality)
            quantiles_mortality[t,:] = np.quantile(mortality, quantiles)

        return (mean_mortality, std_mortality, quantiles_mortality)

    def forecast_mortality(self, age, forecast_length, mc_samples = 1000):

        exposure = self.data[1]

        (level_loc_forecast, level_scale_forecast) = self.forecast_factors(forecast_length)

        mean_mortality_rate = np.empty(forecast_length)
        std_mortality_rate = np.empty(forecast_length)

        for t in range(forecast_length):
            mortality = np.empty(mc_samples)
            for i in range(mc_samples):
                factors = torch.distributions.Normal(level_loc_forecast[:,t] , level_scale_forecast[:,t]).sample()
                intensity = self.dmm.emitter.forward(factors)[age]

            mean_mortality_rate[t] = np.mean(intensity)
            std_mortality_rate[t] = np.std(intensity)

        return (mean_mortality_rate, std_mortality_rate)


    def forecast_factors(self, forecast_length):

        #Distribution of last time in training set
        level_loc = self.level_loc[:,-1]
        trend_loc = self.trend_loc[:,-1]

        level_scale = self.level_scale[:,-1]
        trend_scale = self.trend_scale[:,-1]

        cov = self.level_trend_cov[:,-1]

        #Latent process standard deviation
        level_noise_scale = self.dmm.scale_transform(self.dmm.level_scale)
        trend_noise_scale = self.dmm.scale_transform(self.dmm.trend_scale)

        #Arrays to hold forecast
        level_forecast_loc = torch.zeros((level_loc.size()[0], forecast_length))
        level_forecast_scale = torch.zeros((trend_loc.size()[0], forecast_length))

        for t in range(forecast_length):
            (level_loc, trend_loc) = self.dmm.latent_transition.forward( level_loc, trend_loc )
            (level_scale, trend_scale, cov) = self.dmm.latent_transition.forward_cov( level_scale, trend_scale, cov, level_noise_scale, trend_noise_scale )

            level_forecast_loc[:,t] = level_loc
            level_forecast_scale[:,t] = level_scale
        
        return (level_forecast_loc.detach(), level_forecast_scale.detach())

    def log_score(self, e, d, mc_samples = 1000):

        forecast_length = len(e)

        (fcast_mean, fcast_scale) = self.forecast_factors(forecast_length = forecast_length)

        log_score_t = np.zeros(forecast_length)
        log_score = np.zeros((mc_samples, self.num_ages))

        for t in range(forecast_length):
            for i in range(mc_samples):
                factor_sample = torch.distributions.Normal(fcast_mean[:,t] , fcast_scale[:,t] ).sample()
                intensity = self.dmm.emitter.forward(factor_sample).detach().numpy()
                for age in range(self.num_ages):
                    log_score[i,age] = poisson.logpmf(d[t,age], intensity[age] * e[t,age])
            log_score_t[t] = - np.sum( logsumexp(log_score, axis = 0) - np.log(mc_samples) )
        
        return log_score_t
    
    def log_score_saturated(self, e, d):
        forecast_length = len(e)
        log_score_t = np.empty(forecast_length)
        log_score = np.empty(self.num_ages)

        intensity = d/e

        for t in range(forecast_length):
            for age in range(self.num_ages):
                log_score[age] = poisson.logpmf(d[t,age], intensity[t,age] * e[t,age])
            log_score_t[t] = - np.sum( log_score, axis = 0)

        return log_score_t

    def log_score_naive(self, e, d):
        forecast_length = len(e)
        log_score_t = np.empty(forecast_length)
        log_score = np.empty(self.num_ages)

        intensity = self.data[0][-1,:]/self.data[1][-1,:]

        for t in range(forecast_length):
            for age in range(self.num_ages):
                log_score[age] = poisson.logpmf(d[t,age], intensity[age] * e[t,age])
            log_score_t[t] = - np.sum( log_score, axis = 0)

        return log_score_t