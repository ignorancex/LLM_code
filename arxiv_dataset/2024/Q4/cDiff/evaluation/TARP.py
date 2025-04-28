import torch
from tarp import get_tarp_coverage
import numpy as np
from torch.utils.data import Subset, DataLoader
from datasets.BayesDataStream import BayesDataStream
def ecp_area_difference(ecp, alpha):
    absolute_difference = np.abs(ecp - alpha)
    area_difference = np.trapz(absolute_difference, alpha)
    return area_difference

def get_ecp_area_difference(ds, model, device, n_sim=10000, n_samples=2000):
    """
    Calculate the area difference between the Empirical Coverage Probability (ECP)
    and alpha values based on the provided dataset and model.

    Parameters:
        n_sim: how many pait of (theta,y) is needed
        n_smaple: for each y, how may theta is sampled
    ds (Dataset): The dataset containing input-output pairs.
    n_samples (int): Number of samples to draw for each simulation.
    model (Model): A model that can generate theta samples based on given y values.

    Returns:
    float: The area difference computed using the ECP and alpha values.
    """

    # Select a random sample (theta_star, y) pair from the dataset.

    y_dataset_length = ds.sample_n(1)[0]
    def my_gen_sample_size(n, low=y_dataset_length, high=y_dataset_length + 1):
        return np.random.randint(low=low, high=high, size=n)

    new_ds = BayesDataStream(1, n_sim, ds.sample_theta, ds.sample_y, my_gen_sample_size)
    dl = DataLoader(new_ds, batch_size=n_sim, shuffle=False)
    sample_batch = next(iter(dl))
    theta_star, y = sample_batch
    y = y.to(device)

    # y has shape [n_sim, dataset_num, y_dim]

    # Reshape y to combine n_sim and n_samples into a single batch dimension.
    # Resulting shape is [n_sim * n_samples, dataset_num, y_dim]

    try:
        s = model.summary(y)
        s_expanded = s.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, s.shape[1])
    except:
        s_expanded = y.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(-1, y.shape[1], y.shape[2])
        if s_expanded.shape[1] == 1:
            s_expanded = s_expanded.squeeze(1)

    theta_samples = model.sample_given_s(s_expanded)

    # Reshape theta_samples back to [n_sample, n_sim, dim_theta]
    theta_samples = theta_samples.reshape(n_samples, n_sim, -1)

    # Convert theta samples to numpy array for further processing
    theta_samples = theta_samples.detach().cpu().numpy()

    # Calculate ECP and alpha values using some metric function (assumed to be defined elsewhere)
    ecp, alpha = get_tarp_coverage(theta_samples, theta_star.numpy(), references='random', metric='euclidean', norm=True, seed=5)

    # Calculate the area difference based on ECP and alpha values.
    area = ecp_area_difference(ecp, alpha)

    return area, ecp, alpha



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Step 1: Define prior distribution for mu (Normal distribution)
    mu_0 = 0  # Mean of the prior distribution
    tau_0 = 1  # Standard deviation (sqrt of variance) of the prior distribution

    # Step 2: Sample mu from the prior distribution
    mu_prior = norm.rvs(loc=mu_0, scale=tau_0, size=1)

    # Step 3: Define the likelihood distribution and generate observations based on mu
    sigma = 2  # Standard deviation of the likelihood (observation noise)
    n_samples = 30  # Number of samples to generate from the likelihood
    x = norm.rvs(loc=mu_prior, scale=sigma, size=n_samples)  # Observed data points


    import numpy as np
    from scipy.stats import norm

    n_samples = 20000
    n_sim = 10000

    theta_star = norm.rvs(loc=mu_0, scale=tau_0, size=(n_sim,1)) # shape [n_sim, n_dim]

    x = norm.rvs(loc=theta_star, scale=sigma)
    prec_n = 1 / (tau_0 ** 2) + 1 / (sigma ** 2)
    sigma_n = 1 / prec_n
    mu_n = mu_0 / (tau_0 ** 2) + x / (sigma ** 2)
    mu_n /= prec_n

    # theta_samples = [norm.rvs(loc=mu, scale=sigma_n, size = n_samples) for mu in mu_n]
    theta_samples = [norm.rvs(loc=mu + 100, scale=sigma_n * 10, size=n_samples) for mu in mu_n]
    # theta_samples = [norm.rvs(loc=mu + 1, scale=sigma_n, size=n_samples) for mu in mu_n]
    theta_samples = np.stack(theta_samples) # shape [n_sim, n_samples, n_dim]
    if len(theta_samples.shape) == 2:
        theta_samples = theta_samples[:, :, np.newaxis]
    theta_samples = theta_samples.transpose(1,0,2)

    from tarp import get_tarp_coverage
    ecp, alpha = get_tarp_coverage(theta_samples, theta_star, references='random', metric='euclidean', norm=True, seed=5)
    print(len(ecp), alpha)

    area_difference = ecp_area_difference(ecp, alpha)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.plot([0, 1], [0, 1], ls='--', color='k', label="Ideal case")
    ax.plot(alpha, ecp, label='TARP')
    ax.legend()
    ax.set_ylabel(f"Expected Coverage_{area_difference:.4f}")
    ax.set_xlabel("Credibility Level")

    plt.subplots_adjust(wspace=0.4)
    plt.savefig("trap_bias.png")


