import pyro
import torch
import pyro.distributions as dist



def synthesis_data(nsample):

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nagecat': dist.Categorical(probs=prob_age),
        'UJ': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'UD': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
    }
    S = pyro.sample("Sex", exo_dist['Nsex'])
    R = pyro.sample("Race", exo_dist['Nrace'])
    A = pyro.sample("Age", exo_dist['Nage'])
    AC = pyro.sample("Age_cat", exo_dist['Nagecat'])
    UJ = pyro.sample("UJ", exo_dist['UJ'])
    UD = pyro.sample("UD", exo_dist['UD'])
    expF = torch.exp(R + S + A + AC + UJ)
    expM = torch.exp(R + S + A + AC + UJ)
    expO = torch.exp(R + S + A + AC + UD)
    expP = torch.exp(R + S + A + AC + UD)
    JF = pyro.sample("JF", dist.Poisson(expF))
    JM = pyro.sample("JM", dist.Poisson(expM))
    JO = pyro.sample("JO", dist.Poisson(expO))
    P = pyro.sample("P", dist.Poisson(expP))
    outcome = pyro.sample("P", dist.Bernoulli(torch.sigmoid()))

    return [outcome]

if _name_ == "__main__":
    data = synthesis_data(100000)
