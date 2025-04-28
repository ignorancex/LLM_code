import os
import argparse
from pathlib import Path
import gpflow
from experiments.experiments import RealDataExperiment
from lipschitz_driven_inference.estimators import (
    Dataset,
    NNLipschitzDrivenEstimator,
    Estimator,
    OLS, 
    Sandwich, 
    KDEIW, 
    GLS, 
    GPBCI
)
from typing import Tuple, List, Dict
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt


def parse():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=250)
    parser.add_argument("--num_neighbors", type=int, default=1)
    parser.add_argument("--lipschitz_bound", type=float, default=0.2)
    # List of bandwidths for tree cover experiment
    parser.add_argument("--list_bandwidths", 
                        nargs="+", 
                        default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],  
                        help="List of bandwidths to try (space-separated).")
    parser.add_argument("--num_threads", type=int, default=5)
    parser.add_argument("--seeds_to_plot", type=int, default=5)
    parser.add_argument("--region", type=str, default="west")
    return parser.parse_args()


class TreeCoverExperiment(RealDataExperiment):

    def __init__(
        self,
        name,
        results_dir: Path,
        alpha: float,
        estimators: List[Estimator],
        estimator_kwargs: List[Dict],
        num_seeds: int = 1,
        seeds_to_plot: int = 1,
        parallel_threads: int = 1,
        region: str = "south",
    ):
        super().__init__(
            name=name,
            results_dir=results_dir,
            alpha=alpha,
            estimators=estimators,
            estimator_kwargs=estimator_kwargs,
            num_seeds=num_seeds,
            seeds_to_plot=seeds_to_plot,
            parallel_threads=parallel_threads,
        )
        self.region = region


    def generate_data(
        self,
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:

        # Set the seed
        np.random.seed(seed)

        # Load the tree cover data
        datapath = Path(Path(__file__).parent, "data", self.region)
        S = np.load(Path(datapath, 'S.npy'))
        X = np.load(Path(datapath,'X.npy'))
        Y = np.load(Path(datapath,'y.npy'))
        Sstar = np.load(Path(datapath, 'S_star.npy'))
        Xstar = np.load(Path(datapath,'X_star.npy'))
        Ystar = np.load(Path(datapath,'y_star.npy'))

        # Add trailing dimension for y 
        Y = Y[:, None]
        Ystar = Ystar[:, None]

        # Add an intercept to the covariates
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        Xstar = np.concatenate((np.ones((Xstar.shape[0], 1)), Xstar), axis=1)

        # Subsample the training data to 20% of total data
        n = S.shape[0]
        n_train = int(0.2 * n)
        train_indices = np.random.choice(n, n_train, replace=False)
        S = S[train_indices]
        X = X[train_indices]
        Y = Y[train_indices]

        return Dataset(S, X, Y), Dataset(Sstar, Xstar, Ystar)

    def plot_data(self,
        seed: int = 42,
        ):

        fontsize = 17
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', size=fontsize)
        
        plt.figure(figsize=(15, 6))

        dataset, dataset_star = self.generate_data(seed)
        # For plotting, convert from radians back to latitude longitude
        train_lat = dataset.S[:, 0] * 180 / np.pi
        train_lon = dataset.S[:, 1] * 180 / np.pi
        test_lat = dataset_star.S[:, 0] * 180 / np.pi
        test_lon = dataset_star.S[:, 1] * 180 / np.pi

        plt.scatter(train_lon, train_lat, c="C0", label="Source")
        plt.scatter(test_lon, test_lat, c="C1", label="Target")
        plt.xlabel("Longitude", fontsize=fontsize)
        plt.ylabel("Latitude", fontsize=fontsize)
        # Set fontsize of axis
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        # Generate a meshgrid for plotting the conditional expectation and covariate
        # Save the figure in the results directory   
        os.makedirs(self.results_dir, exist_ok=True)  
        plt.savefig(self.results_dir + "/data_plot.pdf")



if __name__ == "__main__":
    # Parse arguments using argparse
    args = parse()
    # Set up the experiment
    
    file_path = Path(__file__).parent
    results_dir = f"results/real_data/{args.region}"
    results_dir = str(Path(file_path, results_dir))
    experiment = TreeCoverExperiment(
        num_seeds=args.num_seeds,
        seeds_to_plot=args.seeds_to_plot,
        name="tree_cover_experiment",
        results_dir=results_dir,
        alpha=0.05,
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
                "data_on_sphere": True,
            },
            {},
            {},
            {
                "bandwidth": args.list_bandwidths,
            },
            {
                "kernel": gpflow.kernels.Matern32,
                "kernel_kwargs": {"lengthscales": 0.1},
                "nugget_variance": 0.1
            },
            {
                "kernel": gpflow.kernels.Matern32,
                "kernel_kwargs": {"lengthscales": 0.1},
                "nugget_variance": 0.1
            },
        ],
        parallel_threads=args.num_threads,
        region=args.region,
        )
    # Run the experiment
    experiment.plot_data()
    experiment.run()
    # experiment.load_results()
    # Save the results
    experiment.save_results()
    # Plot the results
    experiment.plot_coverage_results(conservative=True)
    experiment.plot_coverage_results(conservative=False)
    experiment.plot_confidence_intervals()

