import numpy as np
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader
import sdeint
import time
import matplotlib.pyplot as plt


# Univariate stochastic volatility model

# Sample theta: AR(1) parameters (mu, phi, sigma_eta)
def sample_theta():
    mu = np.random.normal(-1, 1/2)
    phi = np.random.uniform(0.9, 0.999)  # typically close to 1 for financial data
    sigma_eta = np.random.uniform(0.1, 0.3)
    return {'mu': mu, 'phi': phi, 'sigma_eta': sigma_eta}


# Generate y_t given the AR(1) parameters
# h_t = \mu + phi h_{t-1} + \sigma\eta_t
# dh_t = (1 - \phi)(\mu - h_{t-1})dt + \sigma dW_t

def sample_y_sde(parms, n, t_final=10):
    mu, phi, sigma_eta = parms['mu'], parms['phi'], parms['sigma_eta']

    # Initial value of h from the marginal state distribution
    init_state = mu + np.random.normal(0, sigma_eta / np.sqrt(1 - phi ** 2))

    t = np.linspace(0, t_final, n)

    def f(X, t):
        return (1 - phi) * (mu - X)

    def g(X, t):
        return sigma_eta

    h = sdeint.itoint(f, g, init_state, t)
    # Generate y_t
    epsilon = np.random.normal(0, 1, n)
    y = np.exp(h / 2) * epsilon

    return y.reshape(-1,1)  # to give proper shape


def sample_y_loop(parms, n):
    mu, phi, sigma_eta = parms['mu'], parms['phi'], parms['sigma_eta']

    h = np.zeros(n)
    y = np.zeros(n)

    # Initial value of h from the marginal state distribution

    # I guess here the thing that causes collapse
    # h[0] = mu + np.random.normal(0, sigma_eta / np.sqrt(1 - phi ** 2))
    h[0] = mu + np.random.normal(0, sigma_eta)

    for t in range(1, n):
        h[t] = mu + phi * (h[t - 1] - mu) + np.random.normal(0, sigma_eta)

    # Generate y_t
    epsilon = np.random.normal(0, 1, n)
    y = np.exp(h / 2) * epsilon

    return y.reshape(-1,1)  # to give proper shape

#False can be faster.
def sample_y(parms, n ,sde=False):
    if sde:
        return sample_y_sde(parms, n)
    else:
        return sample_y_loop(parms, n)

# Sample size generator
def my_gen_sample_size(n, low=300, high=1500):
    return np.random.randint(low=low, high=high, size=n)


def return_stochastic_vol_dl(n_batches = 256,batch_size = 128,n_sample=None,return_ds=False):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=300, high=1500):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl


if __name__ == "__main__":
    import sys
    import time

    sys.path.append("..")  # Add the parent directory to the system path
    from my_model import BayesFlowEncoder  # Import the class from the module

    y_dim = 1
    embedding_dim = 64
    encoder = BayesFlowEncoder(y_dim, embedding_dim)

    start_time = time.time()
    dl, ds = return_stochastic_vol_dl(n_batches=2,batch_size=5, return_ds=True, n_sample=500)
    for batch in dl:
        theta, y = batch
        s = encoder(y)
        print(theta.shape)
        print(y.shape)
        print(s.shape)
    print(f"Elps time is {time.time() - start_time} s.")

    theta, y = ds.__getitem__(3)
    print(y.shape)

    y = y[:, 0].numpy()  # Convert to numpy array if it's a tensor

    # Create a time vector
    time_steps = np.arange(len(y))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y,  color='blue')
    plt.title('Stochastic Volume')
    plt.xlabel('Time Steps')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig("../test/sto_vol.png")