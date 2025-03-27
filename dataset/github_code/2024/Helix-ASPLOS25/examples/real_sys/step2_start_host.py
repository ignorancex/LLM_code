# 2024.11.02 Yixuan Mei
import os
import sys

from llm_sys.maxflow_host import run_maxflow_host_online, run_maxflow_host_offline
from llm_sys.heuristic_host import run_heuristic_host_online, run_heuristic_host_offline
from simulator.event_simulator.cluster_simulator import ModelName


def example_maxflow_offline():
    os.makedirs("./result/maxflow_offline/", exist_ok=True)
    print("Running example: maxflow host + offline mode")
    run_maxflow_host_offline(
        # model and machine
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
        model_name=ModelName.LLaMa70B,
        # cluster
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol.ini",
        simulator_cluster_file_name="./layout/simulator_cluster.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        initial_launch_num=20,
        feeding_hwm=0.8,
        # result
        result_logging_dir="./result/maxflow_offline/"
    )


def example_maxflow_online():
    os.makedirs("./result/maxflow_online/", exist_ok=True)
    print("Running example: maxflow host + online mode")
    run_maxflow_host_online(
        # model and machine
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
        model_name=ModelName.LLaMa70B,
        # cluster
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol.ini",
        simulator_cluster_file_name="./layout/simulator_cluster.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        avg_throughput=300,
        # result
        result_logging_dir="./result/maxflow_online/"
    )


def example_heuristic_offline(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/{heuristic}_offline/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host offline
    print(f"Running example: {heuristic} host + offline mode")
    run_heuristic_host_offline(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config.txt",
        initial_launch_num=50,
        duration=300,
        result_logging_dir=result_dir
    )


def example_heuristic_online(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/{heuristic}_online/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host online
    print(f"Running example: {heuristic} host + online mode")
    run_heuristic_host_online(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config.txt",
        avg_throughput=150,
        duration=300,
        result_logging_dir=result_dir
    )


def main():
    # parse arguments
    if len(sys.argv) != 3:
        print("Usage: python3 run_host.py <mode> <scheduling_method>")
        print("  mode: online | offline")
        print("  scheduling_method: maxflow | swarm | random")
        return
    mode = sys.argv[1]
    method = sys.argv[2]

    # run the corresponding example
    if mode == "offline" and method == "maxflow":
        example_maxflow_offline()
    elif mode == "online" and method == "maxflow":
        example_maxflow_online()
    elif mode == "offline" and method in ["swarm", "random"]:
        example_heuristic_offline(method)
    elif mode == "online" and method in ["swarm", "random"]:
        example_heuristic_online(method)
    else:
        print(f"Unsupported mode or scheduling method: [{mode}] [{method}]!")


if __name__ == '__main__':
    main()
