import os
from pathlib import Path
import gpflow
from experiments.experiments import SimulationExperiment
from lipschitz_driven_inference.estimators import (
    Dataset,
    NNLipschitzDrivenEstimator,
    OLS,
    Sandwich,
    KDEIW,
    GLS,
    GPBCI,
)
from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np
from simulation_utils import basic_parser


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

        np.random.seed(seed)
        assert -1 < shift < 1

        def covariate_fn(location: ArrayLike) -> ArrayLike:
            return np.sum(location, axis=-1, keepdims=True)

        def conditional_expectation(
            location: ArrayLike, covariate: ArrayLike
        ) -> ArrayLike:
            return covariate + 0.5 * np.sum(np.square(location), axis=-1, keepdims=True)

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

    def plot_data(
        self,
        seed: int,
        n: int,
        p: int,
        m: int,
        shifts: Tuple[float, float] = [0.5, 0.8],
    ):
        import matplotlib.pyplot as plt

        # latex text, larger

        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("font", size=14)

        fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
        for i, shift in enumerate(shifts):
            assert -1 < shift < 1

            dataset, dataset_star = self.generate_data(seed, n, p, m, 0.0, shift)
            ax[i].scatter(dataset.S[:, 0], dataset.S[:, 1], c="C0", label="Source")
            ax[i].scatter(
                dataset_star.S[:, 0], dataset_star.S[:, 1], c="C1", label="Target"
            )
            ax[i].set_title(f"Shift = {shift}")
        # Generate a meshgrid for plotting the conditional expectation and covariate
        x = np.linspace(-1, 1, 200)
        y = np.linspace(-1, 1, 200)
        X, Y = np.meshgrid(x, y)
        S = np.stack((X, Y), axis=-1)
        S = np.reshape(S, (-1, 2))
        Cov = np.sum(S, axis=-1, keepdims=True)
        F = Cov + 0.5 * np.sum(np.square(S), axis=-1, keepdims=True)
        # Share colorbars for the covariate and conditional expectation

        # Plot the covariate, rasterized to avoid aliasing

        ax[i + 1].scatter(
            S[:, 0],
            S[:, 1],
            c=Cov[:, 0],
            cmap="coolwarm",
            vmin=-2,
            vmax=3,
            rasterized=True,
        )
        ax[i + 1].set_title("Covariate")
        # Plot the conditional expectation
        ax[i + 2].scatter(
            S[:, 0],
            S[:, 1],
            c=F[:, 0],
            cmap="coolwarm",
            vmin=-2,
            vmax=3,
            rasterized=True,
        )
        ax[i + 2].set_title("Expected Response")
        # Add a colorbar to the right of all plots, do this by creating a new axis to the right of existing axes,
        # and adjusting existing axes
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        fig.colorbar(ax[i + 2].collections[0], cax=cbar_ax)
        # Save the figure in the results directory
        os.makedirs(self.results_dir, exist_ok=True)
        plt.savefig(self.results_dir + "/data_plot.pdf")


if __name__ == "__main__":
    # Parse arguments using argparse
    parser = basic_parser()
    # update the parser with the arguments for this experiment
    parser.add_argument("--lipschitz_bound", type=float, default=2 * np.sqrt(2))
    args = parser.parse_args()
    # Set up the experiment
    data_generation_kwargs = {
        "n": args.n,
        "m": args.m,
        "p": 2,
        "noise_std": args.noise_std,
        "include_intercept": True,
    }
    all_shifts = [
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0,
        0.2,
        0.4,
        0.6,
        0.8,
    ]  # range of shift values. Should be in [0, 1]
    file_path = Path(__file__).parent
    results_dir = f"results/two_dim_shift/n={args.n}_p={1}_m={args.m}_noise_std={args.noise_std}"
    results_dir = str(Path(file_path, results_dir))
    experiment = TwoDimensionalShiftExperiment(
        name="two_dim_shift",
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
            GLS,
            GPBCI,
        ],
        estimator_kwargs=[
            {
                "num_neighbors": args.num_neighbors,
                "lipschitz_bound": args.lipschitz_bound,
            },
            {},
            {},
            {
                "bandwidth": args.bandwidth,
            },
            {
                "kernel": gpflow.kernels.Matern32,
                "kernel_kwargs": {"lengthscales": 0.1},
                "nugget_variance": 0.1,
            },
            {
                "kernel": gpflow.kernels.Matern32,
                "kernel_kwargs": {"lengthscales": 0.1},
                "nugget_variance": 0.1,
            },
        ],
        all_shifts=all_shifts,
    )
    # Run the experiment
    experiment.plot_data(
        seed=0,
        n=args.n,
        p=2,
        m=args.m,
        shifts=[0.2, 0.6],
    )
    # Run the experiment
    experiment.run()
    # Save the results
    experiment.save_results()
    # Plot the results
    experiment.plot_results()
