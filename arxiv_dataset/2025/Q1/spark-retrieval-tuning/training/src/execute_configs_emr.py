import argparse
from glob import glob
import logging
import shutil
import pandas as pd
from tqdm import tqdm

from .splits import get_data, get_train_test
from . import optimize_queries_training
from .optimize_queries_training import run_tpcds_execution, run_tpch_execution
from .utils import SPARK_DEFAULTS, exec as sp_exec
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.addHandler(logging.FileHandler(filename='logs/execute_config.log'))
optimize_queries_training.logger = logger

def exec(dataset, size, query_id, conf, execution_id):
    logger.info(f"Executing: {dataset} {size}, {query_id} {execution_id} with conf {conf}")
    # First sleep to avoid any conflict with prior running jobs.
    time.sleep(30)
    exec_time = None
    if dataset == "tpcds":
        # run tpcds query
        exec_time = run_tpcds_execution(execution_id, str(size), str(query_id), conf)
        return exec_time
    elif dataset == "tpch":
        # run tpch query
        exec_time = run_tpch_execution(execution_id, str(size), str(query_id), conf)
        return exec_time
    
    return None

def parse_conf(conf_row):
    return {
            "shuffle_partitions": conf_row['params_spark.sql.shuffle.partitions'],
            "executor_instances": conf_row['params_spark.executor.instances'],
            "executor_memory": conf_row['params_spark.executor.memory'],
            "executor_cores": conf_row['params_spark.executor.cores'],
            "driver_memory": conf_row['params_spark.driver.memory'],
            "driver_cores": conf_row['params_spark.driver.cores']
    }

def run(iterations, dataset, size, local_mode=True):
    optimize_queries_training.LOCAL_MODE = local_mode
    logger.info(f"Running in local mode: {optimize_queries_training.LOCAL_MODE}")
    data = get_data('trials', 'csv', dataset=dataset, size=size)
    _, test = get_train_test(data)
    test_queries = test.groupby('applicationName').first().reset_index()
    
    # generate candidate configurations
    default_conf = {
        "shuffle_partitions": SPARK_DEFAULTS['spark.sql.shuffle.partitions'],
        "executor_instances": SPARK_DEFAULTS['spark.executor.instances'],
        "executor_memory": SPARK_DEFAULTS['spark.executor.memory'],
        "executor_cores": SPARK_DEFAULTS['spark.executor.cores'],
        "driver_memory": SPARK_DEFAULTS['spark.driver.memory'],
        "driver_cores": SPARK_DEFAULTS['spark.driver.cores']
    }
    best_configs = test.sort_values('value').groupby('applicationName').first().reset_index()
    knn_configs = pd.read_csv("src/knn/knn_5_results.csv", header=0)
    sbo_configs = pd.read_csv("src/sbo/sbo_results.csv", header=0)
    simtune_configs = pd.read_csv("src/simtune/test_spark_configs.csv", header=0)
    towardsgeots_configs = pd.read_csv("src/simlearn/neighbours_test_configs.csv", header=0)

    for i in tqdm(range(iterations), desc="Test iteration"):
        for _, query in tqdm(test_queries.iterrows(), desc="Test queries"):
                # run candidate configurations

                logger.info(f"Running default for  {query['applicationName']}")
                default_time = exec(dataset, size, query['query_id'], default_conf, execution_id=f"default_{i}")
                
                logger.info(f"Running knn execution for {query['applicationName']}")
                knn_params = parse_conf(knn_configs[knn_configs['applicationName'] == query['applicationName']].iloc[0])
                knn_time = exec(dataset, size, query['query_id'], knn_params, execution_id=f"knn_{i}")

                logger.info(f"Running sbo execution for {query['applicationName']}")
                sbo_params = parse_conf(sbo_configs[sbo_configs['query'] == query['applicationName']].iloc[0])
                sbo_time = exec(dataset, size, query['query_id'], sbo_params, execution_id=f"sbo_{i}")

                logger.info(f"Running simtune for {query['applicationName']}")
                simtune_params = parse_conf(simtune_configs[simtune_configs['source_query'] == query['applicationName']].iloc[0])
                simtune_time = exec(dataset, size, query['query_id'], simtune_params, execution_id=f"simtune_{i}")

                logger.info(f"Running towardsgeots for {query['applicationName']}")
                towards_params = parse_conf(towardsgeots_configs[towardsgeots_configs['applicationName'] == query['applicationName']].iloc[0])
                towards_time = exec(dataset, size, query['query_id'], towards_params, execution_id=f"towardsgeots_{i}")
                
                logger.info(f"Running optimal for {query['applicationName']}")
                best_params = parse_conf(best_configs[best_configs['applicationName'] == query['applicationName']].iloc[0])
                best_time = exec(dataset, size, query['query_id'], best_params, execution_id=f"best_{i}")

                result = {
                            "query": query['applicationName'],
                            "dataset": dataset,
                            "size": size,
                            "query_id": query["query_id"],
                            "default_exec_time": default_time,
                            "simtune_exec_time": simtune_time,
                            "sbo_exec_time": sbo_time,
                            "towards_geots_time": towards_time,
                            "knn_time": knn_time,
                            "best_exec_time": best_time
                }
                pd.DataFrame([result]).to_csv(f"/home/hadoop/{dataset}_{size}_results.csv", header=False, index=False, mode='a')
                
                for match in glob(f"/home/hadoop/data_{dataset}_*"):
                    shutil.rmtree(match)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for which test set to execute.')
    parser.add_argument('--dataset', choices=['tpch', 'tpcds'])
    parser.add_argument('--size', choices=['100g', '250g', '500g', '750g'])
    parser.add_argument('--iterations', type=int, default=1)

    args = parser.parse_args()
    tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
    run(iterations=args.iterations, dataset=args.dataset, size=args.size)


