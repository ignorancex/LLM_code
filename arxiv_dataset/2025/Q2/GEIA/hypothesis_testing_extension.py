import json
import os
import numpy as np
from scipy import stats


def run_t_test(probs: dict) -> None:
     
    original_mean = np.nanmean(probs["positive_sample_likelihood"])
    negative_mean = np.nanmean(probs["negative_sample_likelihood"])
    original_std = np.nanstd(probs["positive_sample_likelihood"])
    negative_std = np.nanstd(probs["negative_sample_likelihood"])

    print(f"Original mean: {original_mean} with std: {original_std}\nNegative mean: {negative_mean} with std: {negative_std}")

    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(
        probs["positive_sample_likelihood"], 
        probs["negative_sample_likelihood"], 
        nan_policy='omit'
    )

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
            print(f"Stat Sig difference in distributions (p_value={p_value}).")
    else:
        print(f"No stat sig difference in dsitributions (p_value={p_value})")



if __name__ == "__main__":

    log_folder: str = "./logs"
    # Iterate over all files in the folder starting with 'leakage_'
    for file_name in os.listdir(log_folder):
        if file_name.startswith("leakage_") and file_name.endswith(".log"):
            file_path = os.path.join(log_folder, file_name)

            # Read and process the log file
            with open(file_path) as f:
                probs = json.load(f)

            print("----------------------------")
            print(f"Running t-tests for file: {file_name}")
            run_t_test(probs)
            print("-----------------------------")