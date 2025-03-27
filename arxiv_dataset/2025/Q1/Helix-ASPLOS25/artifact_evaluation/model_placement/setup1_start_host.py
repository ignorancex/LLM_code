# 2024.11.02 Yixuan Mei
import os
import sys

from llm_sys.maxflow_host import run_maxflow_host_online, run_maxflow_host_offline
from llm_sys.heuristic_host import run_heuristic_host_online, run_heuristic_host_offline
from simulator.event_simulator.cluster_simulator import ModelName


def main():
    # parse arguments
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <method>"
    method = sys.argv[1]

    if method == "helix":
        print("Running LLaMa 70B + Offline + Helix")
        os.makedirs("./real_sys_results/helix", exist_ok=True)
        run_maxflow_host_offline(
            # model and machine
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
            model_name=ModelName.LLaMa70B,
            # cluster
            complete_cluster_file_name="./config_single/cluster24.ini",
            machine_profile_name="./config_single/machine_profiles.ini",
            # solution
            solution_file_name="./layout_single/ilp/ilp_sol.ini",
            simulator_cluster_file_name="./layout_single/ilp/simulator_cluster.ini",
            real_sys_config_file_name="./layout_single/ilp/real_sys_config.txt",
            # throughput
            duration=300,
            initial_launch_num=20,
            feeding_hwm=0.8,
            # result
            result_logging_dir="./real_sys_results/helix"
        )

    if method == "swarm":
        print("Running LLaMa 70B + Offline + Swarm")
        os.makedirs("./real_sys_results/swarm", exist_ok=True)
        run_maxflow_host_offline(
            # model and machine
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
            model_name=ModelName.LLaMa70B,
            # cluster
            complete_cluster_file_name="./config_single/cluster24.ini",
            machine_profile_name="./config_single/machine_profiles.ini",
            # solution
            solution_file_name="./layout_single/swarm/swarm_sol.ini",
            simulator_cluster_file_name="./layout_single/swarm/simulator_cluster.ini",
            real_sys_config_file_name="./layout_single/swarm/real_sys_config.txt",
            # throughput
            duration=300,
            initial_launch_num=20,
            feeding_hwm=0.8,
            # result
            result_logging_dir="./real_sys_results/swarm"
        )

    if method == "petals":
        print("Running LLaMa 70B + Offline + Petals")
        os.makedirs("./real_sys_results/petals", exist_ok=True)
        run_maxflow_host_offline(
            # model and machine
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
            model_name=ModelName.LLaMa70B,
            # cluster
            complete_cluster_file_name="./config_single/cluster24.ini",
            machine_profile_name="./config_single/machine_profiles.ini",
            # solution
            solution_file_name="./layout_single/petals/petals_sol.ini",
            simulator_cluster_file_name="./layout_single/petals/simulator_cluster.ini",
            real_sys_config_file_name="./layout_single/petals/real_sys_config.txt",
            # throughput
            duration=300,
            initial_launch_num=20,
            feeding_hwm=0.8,
            # result
            result_logging_dir="./real_sys_results/petals"
        )



if __name__ == '__main__':
    main()
