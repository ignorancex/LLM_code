"""
Need to train a meta-learner model to predict the distance between 2 tasks.

To prepare the training set for the meta-learner:
1. Train a workload number of surrogate models using its tuning history that returns runtime (or other perfomance metric) given configuration and task features. will be about (22 + 99)x4 models
2. Generate a random set of configurations
3. Use surrogate models to predict runtime for workloads from the dataset using configuration from the selected random set.
4. Create a table where rows are configuration ids, columns are workloads ids, values are runtime. We compute kendal tau distance on this matrix to find out the distance between workloads.
5. To create features for the training set, we concatenate features for workload 1 with features for workload 2 and computed distance as the target.

Now we are ready to train a regression model to predict the distance.
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import random
from tqdm import tqdm

from ..utils import SPARK_PARAMETER_RANGES

import warnings

warnings.filterwarnings("ignore")

random.seed(0)


def random_config() -> dict:
    config = {
        k.replace("spark.", "params_spark."): random.randrange(
            start=v["low"], stop=v["high"], step=v["step"]
        )
        for k, v in SPARK_PARAMETER_RANGES.items()
    }
    return config


def flatten(xss) -> list:
    return [x for xs in xss for x in xs]


def _train_surrogate_model(name, X, y) -> tuple:
    kernel = 1 * Matern(length_scale=1.0, length_scale_bounds=(1e-9, 1e9))
    # kernel = 1 * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
    # kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    gaussian_process.fit(X, y)

    return (name, gaussian_process)


def train_surrogates(trials: pd.DataFrame, config_columns: list) -> dict:
    workloads = trials.groupby("applicationName")
    models = dict(
        [
            _train_surrogate_model(
                w,
                workloads[config_columns].get_group(w).values,
                workloads["value"].get_group(w).values,
            )
            for w in tqdm(workloads.groups.keys(), "Training surrogate models")
        ]
    )
    return models


def _predict_runtimes(
    models: dict, config_columns: list, number_of_configs: int
) -> dict:
    configs_df = pd.DataFrame([random_config() for i in range(0, number_of_configs)])[
        config_columns
    ]
    runtimes = {}
    for workload, model in tqdm(models.items(), "Predicting runtimes"):
        mean_prediction = model.predict(configs_df.values)
        runtimes[workload] = mean_prediction
    return runtimes


def compute_kendall_tau_distance(
    models: dict, config_columns: list, number_of_configs: int
) -> pd.DataFrame:
    runtimes_df = pd.DataFrame(
        _predict_runtimes(models, config_columns, number_of_configs)
    )
    kendall_coeff = runtimes_df.corr(method="kendall")
    kendall_tau_distance = (1 - kendall_coeff) / 2

    return kendall_tau_distance

def load_workload_features(files):
    features = (
        pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        if isinstance(files, list)
        else pd.read_csv(files)
    )

    trials = features.iloc[
        :,
        features.columns.str.startswith(
            ("applicationName", "applicationId", "value", "params_")
        ),
    ]
    wl_features = features.sort_values('applicationId').groupby("applicationName").first().reset_index()
    wl_features = wl_features.iloc[
        :,
        ~wl_features.columns.str.startswith(("value", "applicationId", "params_")),
    ]
    wl_features.set_index("applicationName", inplace=True, drop=False)

    return (wl_features, trials)


def prepare_features_and_targets(
    files, base_path: str = "src", number_of_configs: int = 1000
):
    wl_features, trials = load_workload_features(files)

    print("Dataset is loaded")

    config_columns = [c for c in trials.columns.values if c.startswith("params_")]

    # scaler = preprocessing.StandardScaler().fit(train_trials[config_columns])
    # with open(f'{base_path}/simlearn/config_scaler.pkl', 'wb') as file:
    # pickle.dump(scaler, file)

    models = train_surrogates(trials, config_columns)

    print("Computing kendall tau distance...")
    kendall_tau_distance = compute_kendall_tau_distance(
        models, config_columns, number_of_configs=number_of_configs
    )

    assert wl_features.index.equals(kendall_tau_distance.index)
    assert kendall_tau_distance.index.equals(kendall_tau_distance.columns)

    print("Saving workload features") 
    wl_features.to_csv(f"{base_path}/simlearn/out/workloads.csv", index=False)

    # Save kendall tau distances
    keep = np.triu(np.ones(kendall_tau_distance.shape), 1).astype("bool")
    kendall_tau_to_csv = kendall_tau_distance.where(keep).stack().reset_index()
    kendall_tau_to_csv.columns = ["applicationName_x", "applicationName_y", "distance"]
    kendall_tau_to_csv.to_csv(f"{base_path}/simlearn/out/kendall_tau_distances.csv", index=False)

    print("Saving kendall tau distances between workloads") 

    return (kendall_tau_distance, wl_features)

if __name__ == "__main__":
    base_path = "src"
    kendall_tau_distance, wl_features = prepare_features_and_targets(
        [f"{base_path}/sbo/features_train.csv"],
        number_of_configs=10000,
        base_path=base_path,
    )
