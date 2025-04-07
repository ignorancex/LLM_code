import subprocess

"""
Constants
"""
TRAINING_DATA_LOCAL_DIR = "data/"

NUMBER_OF_TRIALS = 40

SPARK_DEFAULTS = {
    "spark.sql.shuffle.partitions":200,
    "spark.executor.instances": 4, # our test bed is a 4 node cluster
    "spark.driver.memory": 1, #gb
    "spark.executor.memory": 1, #gb
    "spark.executor.cores": 1,
    "spark.driver.cores": 1
}

MAX_CORES_PER_WORKER = 7
MAX_MEMORY_PER_WORKER = 44
NUM_WORKERS = 4

SPARK_PARAMETER_RANGES = {
    "spark.sql.shuffle.partitions": {
        "name": "spark.sql.shuffle.partitions",
        "low": 50,
        "high": 1000,
        "step": 10
    },
    "spark.executor.instances": {
        "name": "spark.executor.instances",
        "low": 1,
        "high": 28,
        "step": 1
    },
    "spark.driver.memory": {
        "name": "spark.driver.memory",
        "low": 1,
        "high": 44,
        "step": 1
    },
    "spark.executor.memory": {
        "name": "spark.executor.memory",
        "low": 1,
        "high": 44,
        "step": 1
    },
    "spark.executor.cores": {
        "name": "spark.executor.cores",
        "low": 1,
        "high": 7,
        "step": 1
    },
    "spark.driver.cores": {
        "name": "spark.driver.cores",
        "low": 1,
        "high": 7,
        "step": 1
    }
}


"""
Methods
"""
def suggest_spark_configurations(trial):
    """Suggests spark configurations for a given optuna study trial"""

    shuffle_partitions = trial.suggest_int(**SPARK_PARAMETER_RANGES['spark.sql.shuffle.partitions'])
    executor_instances = trial.suggest_int(**SPARK_PARAMETER_RANGES['spark.executor.instances'])
    executor_memory = trial.suggest_int(**SPARK_PARAMETER_RANGES["spark.executor.memory"])
    executor_cores = trial.suggest_int(**SPARK_PARAMETER_RANGES["spark.executor.cores"])
    driver_memory = trial.suggest_int(**SPARK_PARAMETER_RANGES["spark.driver.memory"])
    driver_cores = trial.suggest_int(**SPARK_PARAMETER_RANGES["spark.driver.cores"])

    max_executors_calculated = min([
        (MAX_MEMORY_PER_WORKER // executor_memory) * NUM_WORKERS,
        (MAX_CORES_PER_WORKER // executor_cores) * NUM_WORKERS,
    ])

    constraint = executor_instances - max_executors_calculated # out of bounds when instances > max
    trial.set_user_attr("constraint", [constraint])

    return {
        "shuffle_partitions": shuffle_partitions,
        "executor_instances": min([executor_instances, max_executors_calculated]),
        "executor_memory": executor_memory,
        "executor_cores": executor_cores,
        "driver_memory": driver_memory,
        "driver_cores": driver_cores
    }

def generate_confs_for_spark_submit(confs):
    """
    Takes in a dictionary of spark config values and generates a string to be added to the spark-submit command
    """
    num_executors = f"--num-executors {confs['executor_instances']}"
    executor_memory = f"--executor-memory {confs['executor_memory']}g"
    executor_cores = f"--executor-cores {confs['executor_cores']}"
    shuffle_partitions = f"--conf spark.sql.shuffle.partitions={confs['shuffle_partitions']}"
    driver_memory = f"--driver-memory {confs['driver_memory']}g"
    driver_cores = f"--driver-cores {confs['driver_cores']}"
    return " ".join([
        num_executors, executor_memory, executor_cores, shuffle_partitions, driver_memory, driver_cores
    ])



def exec(command):
    try:
        print("Running command: ", command)
        result = subprocess.run(command, capture_output=True, check=True)
        print(result.stdout)
        return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error
        print("error executing docker command")
        print(e.output)
        print(e.stdout)
        print(e.stderr)
        raise e


def docker_exec(container_id, command):
    try:
        # Execute the Docker exec command
        result = subprocess.run(['docker', 'exec', '-it', container_id] + command,
                                capture_output=True, text=True, check=True)
        # Print the output
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error
        print("error executing docker command")
        print(e.output)
        print(e.stdout)
        print(e.stderr)
        raise e

def get_container_id_from_cluster(cluster_id):
    # docker exec -it $(docker ps -q -f name="tpcds-dist_tpcds") /scripts/download_table_data.sh $data_password tpcds $dataset_size
    try:
        docker_ps_command = ["docker", "ps", "-q", "-f", f"name={cluster_id}"]
        # Run the docker ps command and capture the output
        container_id = subprocess.run(docker_ps_command, capture_output=True, text=True, check=True).stdout.strip()
        return container_id
    
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error
        print("error executing docker command")
        print(e.output)
        print(e.stdout)
        print(e.stderr)
        raise e

def copy_training_data_to_local(container_id, container_data_directory, dest):
    try:
        # Execute the Docker exec command
        result = subprocess.run(['docker', 'cp', f"{container_id}:{container_data_directory}", dest], capture_output=True, check=True, text=True)
        # Print the output
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error
        print("error executing docker command")
        print(e.output)
        print(e.stdout)
        print(e.stderr)
        raise e