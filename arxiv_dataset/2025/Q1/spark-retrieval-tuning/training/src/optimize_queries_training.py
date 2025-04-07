from networkx import constraint
from .utils import *
from time import sleep
from functools import partial

import argparse
import optuna
import logging
import sys

# Example usage
TPCH_CONTAINER_ID = "tpch-spark35_tpch_1"
TPCDS_CONTAINER_ID = "tpcds-spark35_tpcds_1"

TPCH_CLUSTER_CONTAINER_ID = "tpch-dist_tpch"
TPCDS_CLUSTER_CONTAINER_ID = "tpcds-dist_tpcds"

logger: logging.Logger = None

QUERIES = ["{:02d}".format(n) for n in range(1, 23)]

LOCAL_MODE = True

def parse_execution_time(execution_data):
    """
    Parses the execution time out from the execution data collected from running tpch
    """
    return float(execution_data.split("\n")[-2].split("\t")[-1])

def run_tpch_execution(id, size, query, confs):
    """
    Run query, get execution time, and copy over data
    """
    
    updated_confs = generate_confs_for_spark_submit(confs)

    directory_name = f"/home/hadoop/data_tpch_{size}_{query}_{id}"

    # container_id = TPCH_CONTAINER_ID if LOCAL_MODE else get_container_id_from_cluster(TPCH_CLUSTER_CONTAINER_ID)

    # docker_exec(container_id, ['mkdir', directory_name])
    exec(['mkdir', directory_name])
    
    # run query but mark as inf if the execution fails for whatever reason
    try:
        # docker_exec(container_id, ['/scripts/run_tpch.sh', id, size, query, updated_confs])
        exec(['/home/hadoop/scripts/run_tpch.sh', id, size, query, updated_confs])
    except Exception as e:
        logger.exception(e)
        return float('inf')

    sleep(10) # race condition where the listeners may not have finished writing the files after the spark app succeeded and before we copy it

    # get execution time
    # execution_data = docker_exec(container_id, ["cat", f"{directory_name}/tpch_execution_times.txt"])
    execution_data = exec(["cat", f"{directory_name}/tpch_execution_times.txt"])

    
    # copy over metrics and delete them in hdfs
    # copy_training_data_to_local(container_id, directory_name, f"{TRAINING_DATA_LOCAL_DIR}/tpch")
    exec(['cp', '-r', directory_name, f"{TRAINING_DATA_LOCAL_DIR}/tpch"])
    # docker_exec(container_id, ['rm', '-r', directory_name])
    # exec(['rm', '-r', directory_name])

    return parse_execution_time(execution_data)

def run_tpcds_execution(id, size, query, confs):
    """
    Run query, get execution time, and copy over data
    """
    updated_confs = generate_confs_for_spark_submit(confs)

    # logger.info("RUNNING WITH ", id, size, query, confs)

    directory_name = f"/home/hadoop/tpcds-spark/data_tpcds_{size}_{query}_{id}"

    #container_id = TPCDS_CONTAINER_ID if LOCAL_MODE else get_container_id_from_cluster(TPCDS_CLUSTER_CONTAINER_ID)

    #docker_exec(container_id, ['mkdir', directory_name])
    exec(['mkdir', directory_name])

    # run query but mark as inf if the execution fails for whatever reason
    try:
        # docker_exec(container_id, ['/scripts/run_tpcds.sh', id, size, query, updated_confs])
        exec(['/home/hadoop/scripts/run_tpcds.sh', id, size, query, updated_confs])
    except Exception as e:
        logger.exception(e)
        return float('inf')

    sleep(10) # race condition where the listeners may not have finished writing the files after the spark app succeeded and before we copy it

    # get execution time
    # execution_data = docker_exec(container_id, ["cat", f"{directory_name}/../output_result/TIMES.txt"])
    execution_data = exec(["cat", f"{directory_name}/TIMES.txt"])

    
    # copy over metrics and delete them in hdfs
    # copy_training_data_to_local(container_id, directory_name, f"{TRAINING_DATA_LOCAL_DIR}/tpcds")
    exec(['cp', '-r', directory_name, f"{TRAINING_DATA_LOCAL_DIR}/tpcds"])
    # docker_exec(container_id, ['rm', '-r', directory_name])
    #exec(['rm', '-r', directory_name])

    return parse_execution_time(execution_data)


def out_of_bounds_instances(trial):
    """
    Suggestion is out of bounds when the number of instances requested is not possible in the current cluster setup
    """
    return trial.user_attrs["constraint"]


def vanilla_objective_function(trial, id, size, query, dataset):
    """Objective function that runs tpch on container and retrieves execution time"""
    confs = suggest_spark_configurations(trial)

    if dataset == "tpch":
        return run_tpch_execution(f"{id}_{trial.number}", size, query, confs)
    elif dataset == "tpcds":
        return run_tpcds_execution(f"{id}_{trial.number}", size, query, confs)


def optimize_queries(id, size, query, dataset):
    """Main function that runs optimization per query"""

    # create and save study
    study_name = f"{dataset}_{size}_{query}_{id}"
    logger.info(f"Executing optimization of {study_name}")
    db_location = f"sqlite:///{TRAINING_DATA_LOCAL_DIR}/{dataset}/{study_name}.db"
    sampler = optuna.samplers.TPESampler(constraints_func=out_of_bounds_instances)
    study = optuna.create_study(study_name=study_name, storage=db_location, sampler=sampler)
    
    # set initial state
    study.enqueue_trial(SPARK_DEFAULTS)
    
    # run
    study.optimize(partial(vanilla_objective_function, id=id, size=size, query=query, dataset=dataset), n_trials = NUMBER_OF_TRIALS)
    logger.info(f"Finished executing optimization of {study_name}")
    
    # print results
    initial_trial = study.trials[0]
    logger.info(f"Initial execution time: {initial_trial.value}")

    # Print the best trial and its result
    best_trial = study.best_trial
    logger.info(f"Best Trial - Params: {best_trial.params}") 
    logger.info(f"Best execution time: {best_trial.value}")
    
    return


def run(id, size, queries, dataset):
    """Runs and tries to optimize all the queries"""
    logger.info(f"Starting runs for {dataset} with size {size} and id {id}")
    for query in queries:
        optimize_queries(id, size, query, dataset)
    
    logger.info(f"Done running {dataset} for size {size}. Results have an id of {id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for generating training data with a specific dataset.')
    parser.add_argument('--size', type=str, help='The size parameter')
    parser.add_argument('--id', type=str, help='The id parameter')
    parser.add_argument('--queries', type=str, nargs='+', default=QUERIES)
    parser.add_argument('--dataset', type=str, default="tpch", choices=["tpch", "tpcds"])
    parser.add_argument('--mode', type=str, default='local', choices=['local', 'cluster'])

    # Parse the command-line arguments
    args = parser.parse_args()

    LOCAL_MODE = True if args.mode == "local" else False

    # Run
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(f"logs/{args.dataset}_{args.size}_{args.id}.log"))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr.

    logger.info(args.queries)

    run(args.id, args.size, args.queries, args.dataset)
