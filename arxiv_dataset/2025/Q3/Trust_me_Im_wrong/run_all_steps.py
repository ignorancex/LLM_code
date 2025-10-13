"""
This script is used to run all the steps of the pipeline. It is used to create the knowledge dataset and to calculate the uncertainty of the model.
"""

import subprocess
import argparse
import datetime
import json
import os
import subprocess
import sys

print(sys.path)
sys.path.append("~/projects/UncertantyInHallucinations/")
print(sys.path)

if __name__ == "__main__":
    print(
        f"git version {subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()}"
    )
    # print(f"git diff {subprocess.check_output(['git', 'diff']).decode('ascii').strip()}")
    print(f"start time {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    # create yml file of the environment with the date in the name
    os.makedirs("environment/", exist_ok=True)
    os.system(
        f"conda env export > environment/environment_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.yml"
    )
    os.makedirs("results/", exist_ok=True)
    os.makedirs("datasets/", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset_name", type=str, default="triviaqa")
    parser.add_argument("--path_to_datasets", type=str, default="datasets/")
    parser.add_argument("--create_knowledge_dataset", type=bool, default=False)
    parser.add_argument("--uncertainty_calculation", type=bool, default=False)
    parser.add_argument("--k_positive_method", type=str, default="bad_shots")
    parser.add_argument("--run_results", type=bool, default=False)

    if parser.parse_args().create_knowledge_dataset:
        from knowledge_dataset import KnowledgeDataset

        KnowledgeDataset(
            parser.parse_args().model_name, parser.parse_args().path_to_datasets, parser.parse_args().dataset_name
        )
    if parser.parse_args().uncertainty_calculation:
        from uncertainty_calculation import UncertaintyCalculation

        uc = UncertaintyCalculation(
            parser.parse_args().model_name, parser.parse_args().path_to_datasets,
            method_k_positive=parser.parse_args().k_positive_method, dataset_name=parser.parse_args().dataset_name)
        without_results = uc.calculate_probabilities_uncertainty(uc.data_path_know, with_knowledge=True)

    if parser.parse_args().run_results:
        from results import run_results

        run_results()
    print(f"end time {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
