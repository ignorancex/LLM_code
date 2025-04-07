import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from .preprocess import load_workload_features

def topN(arr, n):
    indices = np.argpartition(arr, n - 1)[:n]
    # Extra code if you need the indices in order:
    min_elements = arr[indices]
    min_elements_order = np.argsort(min_elements)
    return indices[min_elements_order]


def topNHistorical(
    model, test_workload_features, historical_workload_features, N=3
) -> pd.DataFrame:
    results = []
    for row in tqdm(range(test_workload_features.shape[0]), f"Finding top {N}"):
        test_workload_name = test_workload_features["applicationName"][row]
        features = test_workload_features.iloc[[row]].merge(
            historical_workload_features, how="cross"
        )
        prediction = model.predict(
            features.drop(["applicationName_x", "applicationName_y"], axis=1)
        )
        top_N = topN(prediction, N)
        for i in top_N:
            results.append(
                (
                    test_workload_name,
                    historical_workload_features["applicationName"].iloc[i],
                    prediction[i],
                )
            )
    return pd.DataFrame(
        results, columns=["applicationName", "neighbour", "distance"]
    )

if __name__ == "__main__":
    base_path = "src"

    # Load model
    with open(f"{base_path}/simlearn/sim_model.pkl", "rb") as pkl_file:
        model = pickle.load(pkl_file)

    # Test similarity between train and test sets from sbo and find top configs 
    test_workload_features_sbo, _ = load_workload_features(
        [f"{base_path}/sbo/features_test.csv"]
    )
    historical_workload_features_sbo, trials = load_workload_features(
        [f"{base_path}/sbo/features_train.csv"]
    )
    neighbours = topNHistorical(
        model, test_workload_features_sbo, historical_workload_features_sbo, N=3
    )

    # merge with best configs
    best_trials = trials.groupby("applicationName")["value"].idxmin()
    best_trials = trials.loc[best_trials]

    neighbours = neighbours.merge(
        best_trials,
        left_on="neighbour",
        right_on="applicationName",
        suffixes=("", "_y"),
    ).drop(["applicationName_y"], axis=1)

    neighbours.to_csv(f"{base_path}/simlearn/neighbours_test_configs.csv", index=False)
