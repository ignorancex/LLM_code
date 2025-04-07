import argparse
import json
import os
import pandas as pd
from utils import NUMBER_OF_TRIALS

UUID_KEY = 'uuid'

def concatenate_json(dataset_name, dataset_size, queries, id, suffix):

    base_path = f"data/{dataset_name}"
    concatenated = []

    for query_number in queries:
        root_name = f"{dataset_name}_{dataset_size}_{query_number}_{id}"

        for trial in range(NUMBER_OF_TRIALS):
            file = f"{base_path}/data_{root_name}_{trial}/{root_name}_{trial}_{suffix}"
            
            # skip failed trials
            if not os.path.exists(file):
                continue
            
            with open(file, 'r') as f:
                trial_data = json.load(f)
                trial_data[0][UUID_KEY] = f"{root_name}_{trial}"

                if suffix == "query_plans.json":
                    # query plan json needs to be updated
                    trial_data[0]['applicationName'] = f"{dataset_name}_{dataset_size}_{query_number}"
                    trial_data[0]['applicationId'] = f"{id}_{trial}"

                concatenated.extend(trial_data)

    filename = f"data/{dataset_name}/{dataset_name}_{dataset_size}_{suffix}"
    exists = os.path.exists(filename)
    
    if exists:
        with open(filename) as file:
            existing = json.load(file) if exists else []
            concatenated.extend(existing)

    with open(filename, 'w') as file:
        json.dump(concatenated, file, indent=4)  # Indent for readability

def concatenate_csv(dataset_name, dataset_size, queries, id, suffix):
    base_path = f"data/{dataset_name}"
    concatenated = pd.DataFrame()

    for query_number in queries:
        root_name = f"{dataset_name}_{dataset_size}_{query_number}_{id}"

        for trial in range(NUMBER_OF_TRIALS):
            file = f"{base_path}/data_{root_name}_{trial}/{root_name}_{trial}_{suffix}"

            if not os.path.exists(file):
                continue

            df = pd.read_csv(file)
            df[UUID_KEY] = f"{root_name}_{trial}"
            df["applicationName"] =  f"{dataset_name}_{dataset_size}_{query_number}"
            df["applicationId"] = f"{id}_{trial}"
            concatenated = pd.concat([concatenated, df], ignore_index=True)
        
    
    concatenated.to_csv(f"data/{dataset_name}/{dataset_name}_{dataset_size}_{suffix}", mode='a')

parser = argparse.ArgumentParser(description="Concatenate dataset files into one file per query")
parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
parser.add_argument("--dataset_size", type=str, help="Size of the dataset")
parser.add_argument("--id", type=str, help="Query ID")

args = parser.parse_args()

dataset_name = args.dataset_name
dataset_size = args.dataset_size

id = args.id

queries = []

if dataset_name == "tpch":
    queries = ["{:02d}".format(i) for i in range(1, 23)]
elif dataset_name == "tpcds":
    queries = ["q{}".format(i) for i in range(1, 100)]
    
concatenate_json(dataset_name, dataset_size, queries, id, "stage_metrics.json")
concatenate_json(dataset_name, dataset_size, queries, id, "task_metrics.json")
concatenate_json(dataset_name, dataset_size, queries, id, "query_plans.json")
concatenate_csv(dataset_name, dataset_size, queries, id, "stage_metrics.csv")
concatenate_csv(dataset_name, dataset_size, queries, id, "task_metrics.csv")