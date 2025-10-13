import numpy as np
import pandas as pd
from utility import run
import os
import argparse

def main():

    parser = argparse.ArgumentParser(description="Run tests on heart, kidney, and mammo datasets with impkKPCA and impk models")

    parser.add_argument('--datasets', nargs='+', choices=["australian", "kidney", "mammo"],
                        default=["australian"],  # Set "heart" as the default dataset
                        help='List of datasets to run tests on (heart, kidney, mammo)')

    parser.add_argument('--models', nargs='+', choices=["HIPMK"],
                        default=["HIPMK"],  # Set "impk" as the default model
                        help='List of models to run tests with (HIPMK, HIPMK_KPCA)')

    parser.add_argument('--missing_types', nargs='+', choices=["mcar", "mar", "mnar"],
                        default=["mcar"],  # Set "mcar" as the default missing type
                        help='List of missing data types to consider (mcar, mar, mnar)')


    parser.add_argument('--missing_rates', nargs='+', type=float, 
                        choices=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Example missing rates
                        default=None,  
                        help='List of missing data rates to consider')

    # Parse the arguments
    args = parser.parse_args()

    datasets = args.datasets
    models = args.models
    missing_types = args.missing_types
    missing_rates = args.missing_rates
    clustering = any(dataset in ["kidney", "mammo"] for dataset in datasets)



    for dataset in datasets:
        # Load data for the current dataset
        path = f"dataset/{dataset}/"
        y = np.load(path + "label.npy")
        # Iterate through each model and missing type
        for missing_type in missing_types:
            for model in models:
                all_results = run(dataset, missing_type, model, missing_rates, y, clustering)
                
                print(all_results)

if __name__ == "__main__":
    main()


