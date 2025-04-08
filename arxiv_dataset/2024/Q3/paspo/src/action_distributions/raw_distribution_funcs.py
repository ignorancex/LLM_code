import torch
from action_distributions.helper_distributions import FixedValueDistribution, ListDistribution, SafeTanhTransform
from action_distributions.stable_beta import StableBeta
from action_distributions.truncated_normal import TruncatedNormal, biased_softplus



def _create_distribution_beta_list(parameters, min, max):
    dists = []
    max = torch.tensor(max, dtype=torch.float32, device=parameters.device)
    min = torch.tensor(min, dtype=torch.float32, device=parameters.device)

    parameters = parameters.exp() #TODO do not hardcode
    
    for p, mi, ma in zip(parameters, min, max):
        #for numerical stability
        if ma - mi < 1e-4 or ma <= 1e-4: #TODO do not hardcode
            #return FixedValueDistribution(min)
            transform = [torch.distributions.AffineTransform(loc=torch.tensor(0, dtype=torch.float32, device=parameters.device), scale=torch.tensor(1, dtype=torch.float32, device=parameters.device), cache_size=1)]

            dists.append(torch.distributions.TransformedDistribution(FixedValueDistribution(mi, parameters.device), transform)) #so that we can have the same code also use the transform
            continue

        param_1 = p[0]
        param_2 = p[1]

        #beta = torch.distributions.Beta(param_1, param_2)
        beta = StableBeta(param_1, param_2, validate_args=False) #for numerical stability
        transform = [torch.distributions.AffineTransform(loc=mi, scale=ma-mi, cache_size=1)]
        distribution = torch.distributions.TransformedDistribution(beta, transform)


        dists.append(distribution)

    return ListDistribution(dists, sum=False)


def _create_distribution_truncated_normal_list(parameters, min, max):
    dists = []
    max = torch.tensor(max, dtype=torch.float32, device=parameters.device)
    min = torch.tensor(min, dtype=torch.float32, device=parameters.device)

    for p, mi, ma in zip(parameters, min, max):
        if ma - mi < 1e-2:
            #return FixedValueDistribution(min)
            transform = [torch.distributions.AffineTransform(loc=torch.tensor(0, dtype=torch.float32, device=parameters.device), scale=torch.tensor(1, dtype=torch.float32, device=parameters.device))]

            dists.append(torch.distributions.TransformedDistribution(FixedValueDistribution(mi, parameters.device), transform))  #so that we can have the same code
            continue

        mu, sigma = p


        upscale = 5.0
        tanh_mu = False

        sigma_clip = False
        sigma_min = -20
        sigma_max = 2

        if sigma_clip:
            sigma = sigma.clamp(sigma_min, sigma_max)

        sigma = sigma.exp()
        #sigma = biased_softplus(1.0,min_val=0.01).forward(sigma)

        #sigma = sigma.clamp_min(1e-3)

        #sigma = 1/(ma-mi) * sigma
        
        if tanh_mu:
            mu = torch.tanh(mu/upscale) * upscale

        #mu = mu + (ma - mi) / 2 + mi
        
        distribution = TruncatedNormal(loc=mu, scale=sigma, a=mi, b=ma)

        transform = [torch.distributions.AffineTransform(loc=0, scale=1)] #same code
        distribution = torch.distributions.TransformedDistribution(distribution, transform)

        dists.append(distribution)

    return ListDistribution(dists, sum=False)


def _create_distribution_gaussian_list(parameters, min, max):
    dists = []
    max = torch.tensor(max, dtype=torch.float32, device=parameters.device)
    min = torch.tensor(min, dtype=torch.float32, device=parameters.device)

    for p, mi, ma in zip(parameters, min, max):
        mu, sigma = p


        if ma - mi < 1e-3:
            #return FixedValueDistribution(min)
            transform = [torch.distributions.AffineTransform(loc=torch.tensor(0, dtype=torch.float32, device=parameters.device), scale=torch.tensor(1, dtype=torch.float32, device=parameters.device))]

            dists.append(torch.distributions.TransformedDistribution(FixedValueDistribution(mi, parameters.device), transform)) #TODO so that we can have the same code
            continue


        #mu, sigma = parameters

        #mu = parameters[..., 0]
        #sigma = parameters[..., 1]

        upscale = 5.0
        tanh_mu = False

        sigma_clip = False
        sigma_min = -20
        sigma_max = 2

        if sigma_clip:
            sigma = sigma.clamp(sigma_min, sigma_max)

        sigma = sigma.exp()
        #sigma = biased_softplus(1.0,min_val=0.01).forward(sigma)

        #sigma = sigma.clamp_min(1e-3)

        #sigma = 1/(ma-mi) * sigma
        
        if tanh_mu:
            mu = torch.tanh(mu/upscale) * upscale

        #mu = mu + (ma - mi) / 2 + mi

        distribution = torch.distributions.Normal(loc=mu, scale=sigma)

        transform = [SafeTanhTransform(), torch.distributions.AffineTransform(loc=(ma + mi)/2, scale=(ma-mi)/2)]
        distribution = torch.distributions.TransformedDistribution(distribution, transform)

        dists.append(distribution)

    return ListDistribution(dists, sum=False)
