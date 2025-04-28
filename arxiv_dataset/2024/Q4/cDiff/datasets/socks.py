import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader

# see https://stackoverflow.com/questions/62454956/parameterization-of-the-negative-binomial-in-scipy-via-mean-and-std
def convert_mu_std_to_r_p(mu, std):
    r = mu ** 2 / (std ** 2 - mu)
    p = 1 - mu / std ** 2
    return r, 1 - p


# Function to simulate pulling socks out of the dryer until the first match is found
def simulate_first_match(socks):
    seen_socks = set()
    for i, sock in enumerate(socks):
        if sock in seen_socks:
            return i + 1  # Return the number of pulls when the first match is found
        seen_socks.add(sock)
    return len(socks)  # In case no match is found (edge case)


# Function to sample from the mixture of two beta distributions
def sample_mixture_beta(alpha1, beta1, alpha2, beta2, mixing_coefficient):
    if np.random.rand() < mixing_coefficient:
        return stats.beta.rvs(alpha1, beta1)
    else:
        return stats.beta.rvs(alpha2, beta2)

# hyper paramaters
K = 10  # Number of dryers
mean_total_socks = 30.
std_total_socks = 15.

def sample_theta():
    # convert to neg binomial parameterization
    r, p = convert_mu_std_to_r_p(mean_total_socks, std_total_socks)

    total = stats.nbinom.rvs(r, p, size=K)
    total = total.astype(np.float64) / float(mean_total_socks)

    return {'total': total}

def sample_y(parms, n):
    total = parms["total"] * float(mean_total_socks)
    K = len(total) #number of dryers
    n_pulls = np.zeros(K, dtype=np.int8)

    # Parameters for the two beta distributions in the mixture
    alpha1, beta1 = 30, 4
    alpha2, beta2 = 50, 50
    mixing_coefficient = 0.75

    # Common proportion of paired socks across all dryers, sampled from the mixture
    prop_paired = sample_mixture_beta(alpha1, beta1, alpha2, beta2, mixing_coefficient)

    # fixed the prop_paired.
    # prop_paired = 0.5

    for k in range(K):
        # Simulate the total number of socks for the k-th dryer
        n_socks = total[k]

        # Calculate the number of pairs and odds based on the common proportion
        n_pairs = int((n_socks * prop_paired) // 2)
        n_odds = n_socks - 2 * n_pairs

        # Generate the population of socks for this dryer
        socks = np.append(np.arange(n_pairs), np.arange(n_pairs + n_odds))

        # Shuffle the socks to simulate pulling them out randomly
        np.random.shuffle(socks)

        # Simulate pulling socks until the first match is found
        first_match_count = simulate_first_match(socks)
        n_pulls[k] = first_match_count
    return n_pulls.reshape(1,-1) / mean_total_socks

def my_gen_sample_size(n, low=1, high=2):
    return np.random.randint(low=low, high=high, size=n)


def return_socks_dl(n_batches = 256,batch_size = 128,n_sample=None, return_ds=False):
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl



if __name__ == "__main__":
    dl, ds = return_socks_dl(n_batches=2,batch_size=20, return_ds=True)
    for batch in dl:
        theta, y = batch
        print(theta.shape)
        print(y.shape)

    theta, y = ds.__getitem__(3)
    print(theta)
    print(y)