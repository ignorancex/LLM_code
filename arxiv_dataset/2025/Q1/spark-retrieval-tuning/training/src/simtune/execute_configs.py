import pandas as pd
import subprocess
from ..optimize_queries_training import run_tpcds_execution, run_tpch_execution
# Executes generated spark configs on the cluster to see if they carry over


# Note: this script was used to generate the simtune data but is deprecated in 
# favour of the one in the parent folder

data_password = input("Enter data password: ")
print("Executing with password: ", data_password)

configs = pd.read_csv("src/simtune/test_spark_configs.csv", header=0)
id = "simtune_test"

# 1. Group by dataset, then size to setup
# 2. run each query and capture result
# 3. teardown
# 4. write results csv

print(configs["source_query"].str.split("_").str)
configs["dataset"] = configs["source_query"].str.split("_").str[0]
configs["size"] = configs["source_query"].str.split("_").str[1]
configs["query_id"] = configs["source_query"].str.split("_").str[2]

datasets = configs["dataset"].unique()
sizes = configs["size"].unique()

results = []

for dataset in datasets:
    for size in sizes:
        if dataset == "tpcds":
            # set up tpcds
            subprocess.call(["./prepare/prepare_tpcds.sh", data_password, size])
        elif dataset == "tpch":
            # setup tpch
            subprocess.call(["./prepare/prepare_tpch.sh", data_password, size])
        else:
            continue

        queries_to_run = configs[(configs["dataset"] == dataset) & (configs["size"] == size)]
        for _, query in queries_to_run.iterrows():
            # run query if tpch or tpcds with parameters
            conf = {
                "shuffle_partitions": query['params_spark.sql.shuffle.partitions'],
                "executor_instances": query['params_spark.executor.instances'],
                "executor_memory": query['params_spark.executor.memory'],
                "executor_cores": query['params_spark.executor.cores'],
                "driver_memory": query['params_spark.driver.memory'],
                "driver_cores": query['params_spark.driver.cores']
            }
            exec_time = None
            if dataset == "tpcds":
                # run tpcds query
                exec_time = run_tpcds_execution(id, size, query['query_id'], conf)
            elif dataset == "tpch":
                # run tpch query
                exec_time = run_tpch_execution(id, size, query["query_id"], conf)
            else:
                continue

            results.append({
                "query": query["source_query"],
                "simtune_runtime": exec_time
            })
            print(results[-1])
        
        if dataset == "tpcds":
            # teardown
            subprocess.call("./teardown/teardown_tpcds.sh")
        elif dataset == "tpch":
            # teardown
            subprocess.call("./teardown/teardown_tpch.sh")
        else:
            continue
    
    pd.DataFrame(results).to_csv("src/simtune/results.csv", index=False)
