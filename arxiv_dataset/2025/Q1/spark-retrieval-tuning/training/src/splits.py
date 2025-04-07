from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import json
import pandas as pd

TPCH_QUERIES = ["{:02d}".format(q) for q in range(1, 23)]
TPCDS_QUERIES = [f"q{q}" for q in range(1, 100)]

seed = 0

TEST_SIZE = 0.1

TPCH_TRAIN, TPCH_TEST = train_test_split(TPCH_QUERIES, test_size=TEST_SIZE, random_state=seed)
TPCDS_TRAIN, TPCDS_TEST = train_test_split(TPCDS_QUERIES, test_size=TEST_SIZE, random_state=seed)

if __name__ == "__main__":
    print("Train tpch: ", TPCH_TRAIN)
    print("Train tpcds: ", TPCDS_TRAIN)
    print("Test tpch: ", TPCH_TEST)
    print("Test tpcds: ", TPCDS_TEST)

def get_train_test(dataset):
    train = dataset[dataset['applicationName'].apply(lambda appName: any([appName.startswith("tpch") and appName.endswith(q) for q in TPCH_TRAIN] + [appName.startswith("tpcds") and appName.endswith(q) for q in TPCDS_TRAIN]))]
    test = dataset[dataset['applicationName'].apply(lambda appName: any([appName.startswith("tpch") and appName.endswith(q) for q in TPCH_TEST] + [appName.startswith("tpcds") and appName.endswith(q) for q in TPCDS_TEST]))]
    return (train, test)

def get_data(data, file_extension, dataset=None, size=None):
    if not dataset:
        dataset = "*"
    if not size:
        size = "*"
    
    files = glob.glob(f"data/emr/{dataset}_{size}_{data}*.{file_extension}")
    print(files)

    if  file_extension == "json":
        concatenated = []
        for f in tqdm(files, desc="Loading json files"):
            data = json.load(open(f, 'r'))
            concatenated.extend(data)
        
        return concatenated
    
    elif file_extension == "csv":
        concatenated = pd.DataFrame()
        for f in tqdm(files, desc="Loading csv files"):
             data = pd.read_csv(f)
             concatenated = pd.concat([concatenated, data], ignore_index=True)
        return concatenated