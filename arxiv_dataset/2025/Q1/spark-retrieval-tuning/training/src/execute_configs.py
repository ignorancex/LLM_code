import argparse
import logging
import pandas as pd
import subprocess
from tqdm import tqdm
from . import optimize_queries_training
from .optimize_queries_training import run_tpcds_execution, run_tpch_execution
from .utils import SPARK_DEFAULTS
import time
# Executes generated spark configs on the cluster to see if they carry over

data_password = input("Enter data password: ")
tqdm.write(f"Executing with password: {data_password}")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.addHandler(logging.FileHandler(filename='logs/execute_config.log'))
optimize_queries_training.logger = logger


def setup_environment(dataset, size):
    if optimize_queries_training.LOCAL_MODE:
        if dataset == "tpcds":
            # set up tpcds
            result = subprocess.run(["./prepare/prepare_tpcds.sh", data_password, size], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info(result)
        elif dataset == "tpch":
            # setup tpch
            result = subprocess.run(["./prepare/prepare_tpch.sh", data_password, size], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info(result)
    else:
        if dataset == "tpcds":
            # set up tpcds
            result = subprocess.run(["./prepare/prepare_tpcds_cluster.sh", data_password, size], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info(result)
        elif dataset == "tpch":
                    # setup tpch
            result = subprocess.run(["./prepare/prepare_tpch_cluster.sh", data_password, size], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info(result)


def teardown_environment(dataset):
    if optimize_queries_training.LOCAL_MODE:
        if dataset == "tpcds":
                    # set up tpcds
            result = subprocess.run("./teardown/teardown_tpcds.sh", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info(result)
        elif dataset == "tpch":
                    # setup tpch
            result = subprocess.run("./teardown/teardown_tpch.sh", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info(result)
    else:
        if dataset == "tpcds":
            # set up tpcds
            result = subprocess.run("./teardown/teardown_tpcds_cluster.sh", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info(result)
        elif dataset == "tpch":
            # setup tpch
            result = subprocess.run("./teardown/teardown_tpch_cluster.sh", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logger.info(result)


def exec(dataset, size, query_id, conf, execution_id):
    logger.info(f"Executing: {dataset} {size}, {query_id} {execution_id} with conf {conf}")
    # First sleep to avoid any conflict with prior running jobs.
    time.sleep(30)
    exec_time = None
    if dataset == "tpcds":
        # run tpcds query
        exec_time = run_tpcds_execution(execution_id, size, query_id, conf)
        return exec_time
    elif dataset == "tpch":
        # run tpch query
        exec_time = run_tpch_execution(execution_id, size, query_id, conf)
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

def knn(output_csv, iterations, local_mode=True):
    optimize_queries_training.LOCAL_MODE = local_mode
    logger.info(f"Running in local mode: {optimize_queries_training.LOCAL_MODE}")

    test_queries = pd.read_csv("data/test_queries.csv", header=0)
    best_configs = pd.read_csv("data/best_test_queries.csv", header=0)
    default_conf = {
        "shuffle_partitions": SPARK_DEFAULTS['spark.sql.shuffle.partitions'],
        "executor_instances": SPARK_DEFAULTS['spark.executor.instances'],
        "executor_memory": SPARK_DEFAULTS['spark.executor.memory'],
        "executor_cores": SPARK_DEFAULTS['spark.executor.cores'],
        "driver_memory": SPARK_DEFAULTS['spark.driver.memory'],
        "driver_cores": SPARK_DEFAULTS['spark.driver.cores']
    }
    knn_configs = pd.read_csv("src/knn/knn_5_results.csv", header=0)
    datasets = test_queries["dataset"].unique()
    # sizes = test_queries["size"].unique()
    sizes = ["100g", "500g"]

    for dataset in tqdm(datasets, desc="Datasets"):
        for size in tqdm(sizes, desc="Sizes"):
            logger.info(f"Setting up cluster for {dataset} {size}")
            setup_environment(dataset, size)
            queries_to_run = test_queries[(test_queries["dataset"] == dataset) & (test_queries["size"] == size)]

            for i in tqdm(range(iterations), desc="Test Set Iterations"):
                id = f"knn_local_{i}"
                for _, query in tqdm(queries_to_run.iterrows(), desc="Queries", total=len(queries_to_run.index)):
                    # run default
                    logger.info(f"Running default for  {query['query']}")
                    default_time = exec(dataset, size, query['query_id'], default_conf, execution_id=f"{id}_default")
                    
                    logger.info(f"Running knn execution for {query['query']}")
                    knn_params = parse_conf(knn_configs[knn_configs['query'] == query['query']].iloc[0])
                    knn_time = exec(dataset, size, query['query_id'], knn_params, execution_id=f"{id}_knn")

                    logger.info(f"Running optimal for {query['query']}")
                    best_params = parse_conf(best_configs[best_configs['query'] == query['query']].iloc[0])
                    best_time = exec(dataset, size, query['query_id'], best_params, execution_id=f"{id}_best")

                    result = {
                        "query": query["query"],
                        "dataset": query["dataset"],
                        "size": query["size"],
                        "query_id": query["query_id"],
                        "default_exec_time": default_time,
                        "knn_exec_time": knn_time,
                        "best_exec_time": best_time
                    }
                    pd.DataFrame([result]).to_csv(output_csv, header=False, index=False, mode='a')
            
            logger.info(f"Tearing down cluster for {dataset} {size}")
            teardown_environment(dataset)


def sbo(output_csv, iterations, local_mode=True):
    optimize_queries_training.LOCAL_MODE = local_mode
    logger.info(f"Running in local mode: {optimize_queries_training.LOCAL_MODE}")

    test_queries = pd.read_csv("data/test_queries.csv", header=0)
    best_configs = pd.read_csv("data/best_test_queries.csv", header=0)
    default_conf = {
        "shuffle_partitions": SPARK_DEFAULTS['spark.sql.shuffle.partitions'],
        "executor_instances": SPARK_DEFAULTS['spark.executor.instances'],
        "executor_memory": SPARK_DEFAULTS['spark.executor.memory'],
        "executor_cores": SPARK_DEFAULTS['spark.executor.cores'],
        "driver_memory": SPARK_DEFAULTS['spark.driver.memory'],
        "driver_cores": SPARK_DEFAULTS['spark.driver.cores']
    }
    sbo_configs = pd.read_csv("src/sbo/sbo_results.csv", header=0)
    datasets = test_queries["dataset"].unique()
    # sizes = test_queries["size"].unique()
    sizes = ["100g", "500g"]

    for dataset in tqdm(datasets, desc="Datasets"):
        for size in tqdm(sizes, desc="Sizes"):
            logger.info(f"Setting up cluster for {dataset} {size}")
            setup_environment(dataset, size)
            queries_to_run = test_queries[(test_queries["dataset"] == dataset) & (test_queries["size"] == size)]

            for i in tqdm(range(iterations), desc="Test Set Iterations"):
                id = f"sbo_local_{i}"
                for _, query in tqdm(queries_to_run.iterrows(), desc="Queries", total=len(queries_to_run.index)):
                    # run default
                    logger.info(f"Running default for  {query['query']}")
                    default_time = exec(dataset, size, query['query_id'], default_conf, execution_id=f"{id}_default")
                    
                    logger.info(f"Running sbo execution for {query['query']}")
                    sbo_params = parse_conf(sbo_configs[sbo_configs['query'] == query['query']].iloc[0])
                    sbo_time = exec(dataset, size, query['query_id'], sbo_params, execution_id=f"{id}_sbo")
                    
                    logger.info(f"Running optimal for {query['query']}")
                    best_params = parse_conf(best_configs[best_configs['query'] == query['query']].iloc[0])
                    best_time = exec(dataset, size, query['query_id'], best_params, execution_id=f"{id}_best")

                    result = {
                        "query": query["query"],
                        "dataset": query["dataset"],
                        "size": query["size"],
                        "query_id": query["query_id"],
                        "default_exec_time": default_time,
                        "sbo_exec_time": sbo_time,
                        "best_exec_time": best_time
                    }
                    pd.DataFrame([result]).to_csv(output_csv, header=False, index=False, mode='a')
            
            logger.info(f"Tearing down cluster for {dataset} {size}")
            teardown_environment(dataset)

def simtune(output_csv, iterations, local_mode=True):
    optimize_queries_training.LOCAL_MODE = local_mode
    logger.info(f"Running in local mode: {optimize_queries_training.LOCAL_MODE}")

    test_queries = pd.read_csv("data/test_queries.csv", header=0)
    best_configs = pd.read_csv("data/best_test_queries.csv", header=0)
    default_conf = {
        "shuffle_partitions": SPARK_DEFAULTS['spark.sql.shuffle.partitions'],
        "executor_instances": SPARK_DEFAULTS['spark.executor.instances'],
        "executor_memory": SPARK_DEFAULTS['spark.executor.memory'],
        "executor_cores": SPARK_DEFAULTS['spark.executor.cores'],
        "driver_memory": SPARK_DEFAULTS['spark.driver.memory'],
        "driver_cores": SPARK_DEFAULTS['spark.driver.cores']
    }
    simtune_configs = pd.read_csv("src/simtune/test_spark_configs.csv", header=0)
    datasets = test_queries["dataset"].unique()
    # sizes = test_queries["size"].unique()
    sizes = ["100g", "500g"]

    for dataset in tqdm(datasets, desc="Datasets"):
        for size in tqdm(sizes, desc="Sizes"):
            logger.info(f"Setting up cluster for {dataset} {size}")
            setup_environment(dataset, size)
            queries_to_run = test_queries[(test_queries["dataset"] == dataset) & (test_queries["size"] == size)]

            for i in tqdm(range(iterations), desc="Test Set Iterations"):
                id = f"simtune_local_{i}"
                for _, query in tqdm(queries_to_run.iterrows(), desc="Queries", total=len(queries_to_run.index)):
                    # run default
                    logger.info(f"Running default for  {query['query']}")
                    default_time = exec(dataset, size, query['query_id'], default_conf, execution_id=f"{id}_default")
                    logger.info(f"Running simtune for {query['query']}")
                    simtune_params = parse_conf(simtune_configs[simtune_configs['source_query'] == query['query']].iloc[0])
                    simtune_time = exec(dataset, size, query['query_id'], simtune_params, execution_id=f"{id}_simtune")
                    logger.info(f"Running optimal for {query['query']}")
                    best_params = parse_conf(best_configs[best_configs['query'] == query['query']].iloc[0])
                    best_time = exec(dataset, size, query['query_id'], best_params, execution_id=f"{id}_best")

                    result = {
                        "query": query["query"],
                        "dataset": query["dataset"],
                        "size": query["size"],
                        "query_id": query["query_id"],
                        "default_exec_time": default_time,
                        "simtune_exec_time": simtune_time,
                        "best_exec_time": best_time
                    }
                    pd.DataFrame([result]).to_csv(output_csv, header=False, index=False, mode='a')
            
            logger.info(f"Tearing down cluster for {dataset} {size}")
            teardown_environment(dataset)

def towardsgeots(output_csv, iterations, local_mode=True):
    optimize_queries_training.LOCAL_MODE = local_mode
    logger.info(f"Running in local mode: {optimize_queries_training.LOCAL_MODE}")

    test_queries = pd.read_csv("data/test_queries.csv", header=0)
    best_configs = pd.read_csv("data/best_test_queries.csv", header=0)
    default_conf = {
        "shuffle_partitions": SPARK_DEFAULTS['spark.sql.shuffle.partitions'],
        "executor_instances": SPARK_DEFAULTS['spark.executor.instances'],
        "executor_memory": SPARK_DEFAULTS['spark.executor.memory'],
        "executor_cores": SPARK_DEFAULTS['spark.executor.cores'],
        "driver_memory": SPARK_DEFAULTS['spark.driver.memory'],
        "driver_cores": SPARK_DEFAULTS['spark.driver.cores']
    }
    towardsgeots_configs = pd.read_csv("src/simlearn/neighbours_test_configs.csv", header=0)
    datasets = test_queries["dataset"].unique()
    # sizes = test_queries["size"].unique()
    sizes = ["100g", "500g"]

    for dataset in tqdm(datasets, desc="Datasets"):
        for size in tqdm(sizes, desc="Sizes"):
            logger.info(f"Setting up cluster for {dataset} {size}")
            setup_environment(dataset, size)
            queries_to_run = test_queries[(test_queries["dataset"] == dataset) & (test_queries["size"] == size)]

            for i in tqdm(range(iterations), desc="Test Set Iterations"):
                id = f"towardsgeots_local_{i}"
                for _, query in tqdm(queries_to_run.iterrows(), desc="Queries", total=len(queries_to_run.index)):
                    # run default
                    logger.info(f"Running default for  {query['query']}")
                    default_time = exec(dataset, size, query['query_id'], default_conf, execution_id=f"{id}_default")
                    logger.info(f"Running towardsgeots for {query['query']}")
                    towards_params = parse_conf(towardsgeots_configs[towardsgeots_configs['applicationName'] == query['query']].iloc[0])
                    towards_time = exec(dataset, size, query['query_id'], towards_params, execution_id=f"{id}_towardsgeots")
                    logger.info(f"Running optimal for {query['query']}")
                    best_params = parse_conf(best_configs[best_configs['query'] == query['query']].iloc[0])
                    best_time = exec(dataset, size, query['query_id'], best_params, execution_id=f"{id}_best")

                    result = {
                        "query": query["query"],
                        "dataset": query["dataset"],
                        "size": query["size"],
                        "query_id": query["query_id"],
                        "default_exec_time": default_time,
                        "towardsgeots_exec_time": towards_time,
                        "best_exec_time": best_time
                    }
                    pd.DataFrame([result]).to_csv(output_csv, header=False, index=False, mode='a')
            
            logger.info(f"Tearing down cluster for {dataset} {size}")
            teardown_environment(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for which test set to execute.')
    parser.add_argument('--experiment', type=str, choices=[
        "simtune",
        "simtune_cluster",
        "sbo",
        "sbo_cluster",
        "knn",
        "knn_cluster",
        "towardsgeots",
        "towardsgeots_cluster"
    ])
    parser.add_argument('--iterations', type=int, default=1)

    args = parser.parse_args()

    if args.experiment == "simtune":
        output_csv = "data/simtune_results_local.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        simtune(output_csv, args.iterations)

    elif args.experiment == "simtune_cluster":
        output_csv = "data/simtune_results_cluster.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        simtune(output_csv, args.iterations, local_mode=False)
    
    if args.experiment == "sbo":
        output_csv = "data/sbo_results_local.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        sbo(output_csv, args.iterations)

    elif args.experiment == "sbo_cluster":
        output_csv = "data/sbo_results_cluster.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        sbo(output_csv, args.iterations, local_mode=False)
    
    if args.experiment == "knn":
        output_csv = "data/knn_results_local.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        knn(output_csv, args.iterations)

    elif args.experiment == "knn_cluster":
        output_csv = "data/knn_results_cluster.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        knn(output_csv, args.iterations, local_mode=False)

    if args.experiment == "towardsgeots":
        output_csv = "data/towardsgeots_results_local.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        towardsgeots(output_csv, args.iterations)

    elif args.experiment == "towardsgeots_cluster":
        output_csv = "data/towardsgeots_results_cluster.csv"
        tqdm.write("Warning, you want to create the csv first if it doesn't exist with just the headers because it's in append mode")
        towardsgeots(output_csv, args.iterations, local_mode=False)
