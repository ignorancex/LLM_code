import os
from pathlib import Path
from experiments.experiments import SimulationExperiment
from lipschitz_driven_inference.estimators import (
    Dataset,
    NNLipschitzDrivenEstimator,
    OLS,
    Sandwich,
    KDEIW,
)
from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np
from simulation_utils import basic_parser


def covariate_fn(location: ArrayLike) -> ArrayLike:
    X1 = np.sin(location[:, 0]) + np.cos(location[:, 1])
    X2 = np.cos(location[:, 0]) - np.sin(location[:, 1])
    X3 = location[:, 0] + location[:, 1]
    return np.stack((X1, X2, X3), axis=-1)

def conditional_expectation(
    location: ArrayLike, covariate: ArrayLike
) -> ArrayLike:
    return covariate[:, 0:1] * covariate[:, 1:2] + 0.5 * np.sum(np.square(location), axis=-1, keepdims=True)


class TwoDimensionalShiftExperiment(SimulationExperiment):

    def generate_data(
        self,
        seed: int,
        n: int,
        p: int,
        m: int,
        noise_std: float,
        shift: float = 0,
        include_intercept: bool = False,
    ) -> Tuple[Dataset, Dataset]:

        assert -1 < shift < 1
        S = 2 * np.random.rand(n, 2) - 1
        X = covariate_fn(S)
        F = conditional_expectation(S, X)
        Y = F + noise_std * np.random.randn(n, 1)

        Sstar = ((2 * np.random.rand(m, 2) - 1) + shift) / (1 + np.abs(shift))
        Xstar = covariate_fn(Sstar)
        Fstar = conditional_expectation(Sstar, Xstar)
        Ystar = Fstar + noise_std * np.random.randn(m, 1)

        if include_intercept:
            X = np.concatenate((np.ones((n, 1)), X), axis=1)
            Xstar = np.concatenate((np.ones((m, 1)), Xstar), axis=1)

        return Dataset(S, X, Y), Dataset(Sstar, Xstar, Fstar)

    def plot_data(self,
        seed: int,
        n: int,
        p: int,
        m: int,
        shifts: Tuple[float, float] = [0.5, 0.8],
        ):
        import matplotlib.pyplot as plt
        # latex text, larger

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', size=14)

        fig, ax = plt.subplots(1, 4, figsize=(14, 3), sharey=True)

        # Generate a meshgrid for plotting the conditional expectation and covariate
        x = np.linspace(-1, 1, 200)
        y = np.linspace(-1, 1, 200)
        X, Y = np.meshgrid(x, y)
        S = np.stack((X, Y), axis=-1)
        S = np.reshape(S, (-1, 2))
        Cov = covariate_fn(S)
        F = conditional_expectation(S, Cov)
        # Share colorbars for the covariate and conditional expectation

        # Plot the covariate, rasterized to avoid aliasing
        i = 0
        ax[i].scatter(S[:, 0], S[:, 1], c=Cov[:, 0], cmap="coolwarm", vmin=-2, vmax=3, rasterized=True)
        ax[i].set_title("Covariate 1")
        ax[i+1].scatter(S[:, 0], S[:, 1], c=Cov[:, 1], cmap="coolwarm", vmin=-2, vmax=3, rasterized=True)
        ax[i+1].set_title("Covariate 2")
        ax[i+2].scatter(S[:, 0], S[:, 1], c=Cov[:, 2], cmap="coolwarm", vmin=-2, vmax=3, rasterized=True)
        ax[i+2].set_title("Covariate 3")
        # Plot the conditional expectation
        ax[-1].scatter(S[:, 0], S[:, 1], c=F[:, 0], cmap="coolwarm", vmin=-2, vmax=3, rasterized=True)
        ax[-1].set_title("Expected Response")
        # Add a colorbar to the right of all plots, do this by creating a new axis to the right of existing axes,
        # and adjusting existing axes
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        fig.colorbar(ax[-1].collections[0], cax=cbar_ax)  
        # Save the figure in the results directory   
        os.makedirs(self.results_dir, exist_ok=True)  
        plt.savefig(self.results_dir + "/data_plot.pdf")



if __name__ == "__main__":
    # Parse arguments using argparse
    parser = basic_parser()
    # update the parser with the arguments for this experiment
    parser.add_argument("--lipschitz_bound", type=float, default=3 * np.sqrt(2))
    args = parser.parse_args()
    # Set up the experiment
    data_generation_kwargs = {
        "n": args.n,
        "m": args.m,
        "p": 4,
        "noise_std": args.noise_std,
        "include_intercept": True,
    }
    all_shifts = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]  # range of shift values. Should be in [0, 1]
    file_path = Path(__file__).parent
    results_dir = f"results/two_dim_shift_threecov/n={args.n}_p={3}_m={args.m}_noise_std={args.noise_std}"
    results_dir = str(Path(file_path, results_dir))
    experiment = TwoDimensionalShiftExperiment(
        name="two_dim_shift_threecov",
        results_dir=results_dir,
        dim=1,  # dimension of interest. This is X_1 since there is an intercept
        num_seeds=args.num_seeds,
        alpha=0.05,
        parallel_threads=args.num_threads,
        data_generation_kwargs=data_generation_kwargs,
        estimators=[
            NNLipschitzDrivenEstimator,
            OLS,
            Sandwich,
            KDEIW,
        ],
        estimator_kwargs=[
            {
                "num_neighbors": args.num_neighbors,
                "lipschitz_bound": args.lipschitz_bound,
                "fast_noise": True,
            },
            {},
            {},
            {
                "bandwidth": args.bandwidth,
            },
        ],
        all_shifts=all_shifts,
    )
    # Run the experiment
    experiment.plot_data(
        seed=0,
        n=args.n,
        p=4,
        m=args.m,
        shifts=[-0.4, 0.6],
    )
    # Run the experiment
    experiment.run()
    # Save the results
    experiment.save_results()
    # Plot the results
    experiment.plot_results()