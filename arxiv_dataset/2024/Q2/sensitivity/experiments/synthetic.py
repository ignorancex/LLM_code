import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.stats import multivariate_normal
from primary_sensitivity import primary_sensitivity as ps
from utils import *

import matplotlib as mpl
import seaborn as sns

sns.set_theme()
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
plt.rcParams["xtick.major.pad"] = -1
plt.rcParams["ytick.major.pad"] = -1
plt.rcParams["axes.labelpad"] = 13


def generate_uniform(n):
    return np.random.uniform(-10, 10, (n, 2))


def generate_bivariate_gaussian(n):
    mean = [0, 0]
    # ensure a positive definite covariance matrix
    A = np.random.rand(2, 2)
    covariance = np.dot(
        A, A.transpose()
    )  # A * A^T will be symmetric and positive definite
    return np.random.multivariate_normal(mean, covariance, size=n)


def generate_bivariate_dirichlet(n):
    alpha = np.random.rand(3) * 10  # random parameters greater than 0
    samples = np.random.dirichlet(alpha, size=n)
    return samples[:, :2]  # ignore the third variable


def generate_gmm(n):
    num_components = np.random.randint(
        1, 5
    )  # randomly choose between 1 and 4 components
    gmm = GaussianMixture(n_components=num_components)
    data, _ = gmm.sample(n_samples=n)
    return data


def generate_bivariate_gaussian_outliers(n):
    mean = [0, 0]
    covariance = [[1, 0.5], [0.5, 1]]
    data = np.random.multivariate_normal(mean, covariance, size=n)
    # add outliers
    num_outliers = int(0.1 * n)  # 10% as outliers
    outliers = np.random.uniform(-30, 30, (num_outliers, 2))
    data[:num_outliers] = outliers
    return data


def test_data(data, x_range=None, y_range=None):
    if x_range is None or y_range is None:
        x_range = (data[0, :].min(), data[0, :].max())
        y_range = (data[1, :].min(), data[1, :].max())
    data = data.tolist()
    bounds = ((x_range[0], y_range[0]), (x_range[1], y_range[1]))
    prd = ps(data, bounds)
    brute = brute_force_search(data, bounds)
    return prd, brute


def main():
    np.random.seed(123)
    niter = 500

    df = pd.DataFrame()
    all_data = []

    for i in tqdm(range(niter)):
        for N in [10, 50, 100]:
            data = generate_uniform(N)
            prd, brute = test_data(data)
            all_data.append(dict(N=N, model="uniform", prd=prd[0], brute=brute[0]))

            data = generate_bivariate_gaussian(N)
            prd, brute = test_data(data)
            all_data.append(dict(N=N, model="gaussian", prd=prd[0], brute=brute[0]))

            data = generate_bivariate_dirichlet(N)
            prd, brute = test_data(data)
            all_data.append(dict(N=N, model="dirichlet", prd=prd[0], brute=brute[0]))

            data = generate_bivariate_gaussian_outliers(N)
            prd, brute = test_data(data)
            all_data.append(dict(N=N, model="outliers", prd=prd[0], brute=brute[0]))

    df = pd.DataFrame(all_data)
    prd = df["prd"]
    brute = df["brute"]
    mask = np.isclose(prd, brute)
    print("Fraction accurately predicted:", mask.sum() / len(mask))

    fig, axs = plt.subplots(3, 4, figsize=(5.5, 4), sharex=True, sharey=True)
    axs = axs.ravel()

    for i, ((N, model), group) in enumerate(df.groupby(["N", "model"])):
        axs[i].plot(group["brute"], group["prd"], ".")
        axs[i].plot(np.linspace(0, 2, 100), np.linspace(0, 2, 100), "--r")
        axs[i].grid(False)
        axs[i].set_xlim(0, 2)
        axs[i].set_ylim(0, 2)
        if i < 4:
            axs[i].set_title(model.capitalize())
        if i % 4 == 3:
            axs[i].yaxis.set_label_position("right")
            axs[i].set_ylabel("$N=$" + str(N), rotation=-90)

    fig.text(0.5, 0.02, s="Brute-force", ha="center", va="center")
    fig.text(
        0.02, 0.5, s="Predicted", ha="center", va="center", rotation=90
    )

    plt.tight_layout(h_pad=0.7, w_pad=0.6, pad=1.15)
    plt.savefig("synthetic_plot.pdf")


if __name__ == "__main__":
    main()
