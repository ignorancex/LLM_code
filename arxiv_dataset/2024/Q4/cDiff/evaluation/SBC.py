import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance


def sample_sbc_calstats(ds, n_cal, L, y_dim, model, DEVICE):
    sbc_calstats = torch.empty(size=(n_cal, y_dim))
    for i in tqdm(range(n_cal)):
        theta, y = ds[np.random.randint(low=0, high=ds.__len__())]
        theta = theta.to(DEVICE)
        y = y.repeat(L, 1, 1).to(DEVICE)

        if y.shape[1] == 1:
            y = y.squeeze(1)
        theta_draw = model.sample(y)

        # check rank statistic of theta with respect to the posterior draws
        out = (theta_draw < theta).float().mean(dim=0)
        sbc_calstats[i, :] = out
    return sbc_calstats


def evaluate_sbc(sbc_calstats, seed, epoch_index, model_type):
    num_columns = sbc_calstats.shape[1]
    evaluation_df = pd.DataFrame()

    for i in range(num_columns):
        # Calculate the distance for the i-th column
        distance = compare_distributions(sbc_calstats[:, i])

        # Convert the distance dictionary to a DataFrame
        _df = pd.DataFrame([distance])
        _df["seed"] = seed
        _df["epochs"] = epoch_index
        _df["model_type"] = model_type
        _df["theta_index"] = i

        # Append the _df DataFrame to evaluation_df
        evaluation_df = pd.concat([evaluation_df, _df], ignore_index=True)

    return evaluation_df


def compare_distributions(data, num_points=1000):
    # Kernel density estimation (KDE) for the empirical distribution
    try:
        kde = gaussian_kde(data, bw_method='scott')
    except np.linalg.LinAlgError as e:
        print("Data that caused the issue:")
        print(data)

    # Define the uniform distribution's PDF over [0, 1]
    uniform_pdf = lambda x: 1.0 * np.logical_and(x >= 0, x <= 1)

    # Define a range over which to evaluate the distributions
    x_values = np.linspace(0, 1, num_points)

    # Evaluate the PDFs
    p = kde(x_values)
    q = uniform_pdf(x_values)

    #total_variation_distance = 0.5 * np.sum(np.abs(p - q)) * (x_values[1] - x_values[0])
    total_variation_distance = total_variation(data)

    # Hellinger Distance
    hel_distance = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2) * (x_values[1] - x_values[0]))

    # Wasserstein Distance
    # For Wasserstein, we use the empirical CDFs
    empirical_cdf = np.cumsum(p) * (x_values[1] - x_values[0])
    uniform_cdf = np.cumsum(q) * (x_values[1] - x_values[0])
    wasserstein_dist = wasserstein_distance(empirical_cdf, uniform_cdf)

    return {"tv":total_variation_distance,
            "hl":hel_distance,
            "wa":wasserstein_dist}


def empirical_cdf(samples):
    sorted_samples = np.sort(samples)
    cdf = np.arange(1, len(samples) + 1) / len(samples)
    return sorted_samples, cdf

def uniform_cdf(samples):
    sorted_samples = np.sort(samples)
    uniform_cdf = sorted_samples / sorted_samples[-1]
    return uniform_cdf

def total_variation(samples):
    sorted_samples, empirical_cdf_values = empirical_cdf(samples)
    uniform_cdf_values = uniform_cdf(sorted_samples)
    tv = np.max(np.abs(empirical_cdf_values - uniform_cdf_values))
    return tv