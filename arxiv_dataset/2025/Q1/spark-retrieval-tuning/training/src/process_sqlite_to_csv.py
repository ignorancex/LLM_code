import optuna
import pandas as pd
import os

def get_trials_data(dataset, size, query, id):
    study_name = f"{dataset}_{size}_{query}_{id}"
    database_file = f"data/{dataset}/{study_name}.db"
    if os.path.exists(database_file):
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{database_file}")
        df = study.trials_dataframe()
        df['applicationName'] = f"{dataset}_{size}_{query}"
        df['applicationId'] = df['number'].apply(lambda n: f"{id}_{n}")
        return df
    else:
        #print(f"{database_file} does not exist!")
        return None

tpch = [get_trials_data("tpch", size, f"{query:02d}", "pqt") for size in ["1g", "10g", "100g", "500g"] for query in range(1, 23)]
tpch = [t for t in tpch if t is not None]

tpcds = [get_trials_data("tpcds", size, f"q{query}", id) for size in ["1g", "10g", "100g", "500g"] for query in range(1, 100) ]
tpcds = [t for t in tpcds if t is not None]
print(len(tpcds))

dfs = tpch + tpcds
print(len(dfs))

dataset = pd.concat(dfs , ignore_index=True)

dataset = dataset[[
    "applicationName", 
    "applicationId", 
    "value", 
    "params_spark.driver.cores", 
    "params_spark.executor.memory", 
    "params_spark.executor.cores",
    "params_spark.sql.shuffle.partitions",
    "params_spark.executor.instances",
    "params_spark.driver.memory"
]]

dataset.to_csv("data/trials.csv", index=False)