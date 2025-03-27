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
    assert len(sys.argv) == 4, f"Usage: python {sys.argv[0]} <method> <llama30b/llama70b> <online/offline>"
    method, model_name, serving_mode = sys.argv[1], sys.argv[2], sys.argv[3]
    assert model_name in ["llama30b", "llama70b"], f"Invalid model name: {model_name}"
    assert serving_mode in ["online", "offline"], f"Invalid serving mode: {serving_mode}"

    if model_name == "llama30b" and serving_mode == "offline" and method == "helix_a100":
        print("Running LLaMa 30B + Offline + Helix (A100)")
        os.makedirs("./real_llama30b/helix_offline/a100", exist_ok=True)
        run_maxflow_host_offline(
            # model and machine
            machine_num_dict={"A100": 4},
            model_name=ModelName.LLaMa30B,
            # cluster
            complete_cluster_file_name="./config/a100.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama30b/ilp/a100/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/a100/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama30b/ilp/a100/real_sys_config.txt",
            # throughput
            duration=300,
            initial_launch_num=20,
            feeding_hwm=0.8,
            # result
            result_logging_dir="./real_llama30b/helix_offline/a100"
        )

    if model_name == "llama30b" and serving_mode == "offline" and method == "helix_l4":
        print("Running LLaMa 30B + Offline + Helix (L4)")
        os.makedirs("./real_llama30b/helix_offline/l4", exist_ok=True)
        run_maxflow_host_offline(
            # model and machine
            machine_num_dict={"L4": 8},
            model_name=ModelName.LLaMa30B,
            # cluster
            complete_cluster_file_name="./config/l4.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama30b/ilp/l4/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/l4/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama30b/ilp/l4/real_sys_config.txt",
            # throughput
            duration=300,
            initial_launch_num=20,
            feeding_hwm=0.8,
            # result
            result_logging_dir="./real_llama30b/helix_offline/l4"
        )

    if model_name == "llama30b" and serving_mode == "offline" and method == "helix_t4":
        print("Running LLaMa 30B + Offline + Helix (T4)")
        os.makedirs("./real_llama30b/helix_offline/t4", exist_ok=True)
        run_maxflow_host_offline(
            # model and machine
            machine_num_dict={"T4": 12},
            model_name=ModelName.LLaMa30B,
            # cluster
            complete_cluster_file_name="./config/t4.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama30b/ilp/t4/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/t4/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama30b/ilp/t4/real_sys_config.txt",
            # throughput
            duration=300,
            initial_launch_num=20,
            feeding_hwm=0.8,
            # result
            result_logging_dir="./real_llama30b/helix_offline/t4"
        )

    if model_name == "llama30b" and serving_mode == "online" and method == "helix_a100":
        print("Running LLaMa 30B + Online + Helix (A100)")
        os.makedirs("./real_llama30b/helix_online/a100", exist_ok=True)
        run_maxflow_host_online(
            # model and machine
            machine_num_dict={"A100": 4},
            model_name=ModelName.LLaMa30B,
            # cluster
            complete_cluster_file_name="./config/a100.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama30b/ilp/a100/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/a100/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama30b/ilp/a100/real_sys_config.txt",
            # throughput
            duration=300,
            avg_throughput=600,
            # result
            result_logging_dir="./real_llama30b/helix_online/a100"
        )

    if model_name == "llama30b" and serving_mode == "online" and method == "helix_l4":
        print("Running LLaMa 30B + Online + Helix (L4)")
        os.makedirs("./real_llama30b/helix_online/l4", exist_ok=True)
        run_maxflow_host_online(
            # model and machine
            machine_num_dict={"L4": 8},
            model_name=ModelName.LLaMa30B,
            # cluster
            complete_cluster_file_name="./config/l4.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama30b/ilp/l4/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/l4/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama30b/ilp/l4/real_sys_config.txt",
            # throughput
            duration=300,
            avg_throughput=200,
            # result
            result_logging_dir="./real_llama30b/helix_online/l4"
        )

    if model_name == "llama30b" and serving_mode == "online" and method == "helix_t4":
        print("Running LLaMa 30B + Online + Helix (T4)")
        os.makedirs("./real_llama30b/helix_online/t4", exist_ok=True)
        run_maxflow_host_online(
            # model and machine
            machine_num_dict={"T4": 12},
            model_name=ModelName.LLaMa30B,
            # cluster
            complete_cluster_file_name="./config/t4.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama30b/ilp/t4/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/t4/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama30b/ilp/t4/real_sys_config.txt",
            # throughput
            duration=300,
            avg_throughput=170,
            # result
            result_logging_dir="./real_llama30b/helix_online/t4"
        )

    if model_name == "llama30b" and serving_mode == "offline" and method == "swarm":
        print("Running LLaMa 30B + Offline + Swarm")
        os.makedirs("./real_llama30b/swarm_offline", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="swarm",
            real_sys_config_file_name="./layout_llama30b/swarm/real_sys_config.txt",
            initial_launch_num=50,
            duration=300,
            result_logging_dir="./real_llama30b/swarm_offline"
        )

    if model_name == "llama30b" and serving_mode == "online" and method == "swarm":
        print("Running LLaMa 30B + Online + Swarm")
        os.makedirs("./real_llama30b/swarm_online", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="swarm",
            real_sys_config_file_name="./layout_llama30b/swarm/real_sys_config.txt",
            avg_throughput=450,
            duration=300,
            result_logging_dir="./real_llama30b/swarm_online"
        )

    if model_name == "llama30b" and serving_mode == "offline" and method == "separate_a100":
        print("Running LLaMa 30B + Offline + Separate (A100)")
        os.makedirs("./real_llama30b/separate_offline/a100", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama30b/separate/a100/real_sys_config.txt",
            initial_launch_num=30,
            duration=300,
            result_logging_dir="./real_llama30b/separate_offline/a100"
        )

    if model_name == "llama30b" and serving_mode == "offline" and method == "separate_l4":
        print("Running LLaMa 30B + Offline + Separate (L4)")
        os.makedirs("./real_llama30b/separate_offline/l4", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama30b/separate/l4/real_sys_config.txt",
            initial_launch_num=30,
            duration=300,
            result_logging_dir="./real_llama30b/separate_offline/l4"
        )

    if model_name == "llama30b" and serving_mode == "offline" and method == "separate_t4":
        print("Running LLaMa 30B + Offline + Separate (T4)")
        os.makedirs("./real_llama30b/separate_offline/t4", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama30b/separate/t4/real_sys_config.txt",
            initial_launch_num=30,
            duration=300,
            result_logging_dir="./real_llama30b/separate_offline/t4"
        )

    if model_name == "llama30b" and serving_mode == "online" and method == "separate_a100":
        print("Running LLaMa 30B + Online + Separate (A100)")
        os.makedirs("./real_llama30b/separate_online/a100", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama30b/separate/a100/real_sys_config.txt",
            avg_throughput=500,
            duration=300,
            result_logging_dir="./real_llama30b/separate_online/a100"
        )

    if model_name == "llama30b" and serving_mode == "online" and method == "separate_l4":
        print("Running LLaMa 30B + Online + Separate (L4)")
        os.makedirs("./real_llama30b/separate_online/l4", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama30b/separate/l4/real_sys_config.txt",
            avg_throughput=150,
            duration=300,
            result_logging_dir="./real_llama30b/separate_online/l4"
        )

    if model_name == "llama30b" and serving_mode == "online" and method == "separate_t4":
        print("Running LLaMa 30B + Online + Separate (T4)")
        os.makedirs("./real_llama30b/separate_online/t4", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama30b/separate/t4/real_sys_config.txt",
            avg_throughput=150,
            duration=300,
            result_logging_dir="./real_llama30b/separate_online/t4"
        )

    if model_name == "llama70b" and serving_mode == "offline" and method == "helix":
        print("Running LLaMa 70B + Offline + Helix")
        os.makedirs("./real_llama70b/helix_offline", exist_ok=True)
        run_maxflow_host_offline(
            # model and machine
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
            model_name=ModelName.LLaMa70B,
            # cluster
            complete_cluster_file_name="./config/cluster24.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama70b/ilp/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama70b/ilp/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama70b/ilp/real_sys_config.txt",
            # throughput
            duration=300,
            initial_launch_num=20,
            feeding_hwm=0.8,
            # result
            result_logging_dir="./real_llama70b/helix_offline"
        )

    if model_name == "llama70b" and serving_mode == "online" and method == "helix":
        print("Running LLaMa 70B + Online + Helix")
        os.makedirs("./real_llama70b/helix_online", exist_ok=True)
        run_maxflow_host_online(
            # model and machine
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
            model_name=ModelName.LLaMa70B,
            # cluster
            complete_cluster_file_name="./config/cluster24.ini",
            machine_profile_name="./config/machine_profiles.ini",
            # solution
            solution_file_name="./layout_llama70b/ilp/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama70b/ilp/simulator_cluster.ini",
            real_sys_config_file_name="./layout_llama70b/ilp/real_sys_config.txt",
            # throughput
            duration=300,
            avg_throughput=1400,  # change this to 700 if you want to run 30 minutes
            # result
            result_logging_dir="./real_llama70b/helix_online"
        )

    if model_name == "llama70b" and serving_mode == "offline" and method == "swarm":
        print("Running LLaMa 70B + Offline + Swarm")
        os.makedirs("./real_llama70b/swarm_offline", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="swarm",
            real_sys_config_file_name="./layout_llama70b/swarm/real_sys_config.txt",
            initial_launch_num=180,
            duration=300,
            result_logging_dir="./real_llama70b/swarm_offline"
        )

    if model_name == "llama70b" and serving_mode == "online" and method == "swarm":
        print("Running LLaMa 70B + Online + Swarm")
        os.makedirs("./real_llama70b/swarm_online", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="swarm",
            real_sys_config_file_name="./layout_llama70b/swarm/real_sys_config.txt",
            avg_throughput=700,
            duration=300,
            result_logging_dir="./real_llama70b/swarm_online"
        )

    if model_name == "llama70b" and serving_mode == "offline" and method == "separate_a100":
        print("Running LLaMa 70B + Offline + Separate (A100)")
        os.makedirs("./real_llama70b/separate_offline/a100", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama70b/separate/real_sys_config_a100.txt",
            initial_launch_num=18,
            duration=300,
            result_logging_dir="./real_llama70b/separate_offline/a100"
        )

    if model_name == "llama70b" and serving_mode == "offline" and method == "separate_l4":
        print("Running LLaMa 70B + Offline + Separate (L4)")
        os.makedirs("./real_llama70b/separate_offline/l4", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama70b/separate/real_sys_config_l4.txt",
            initial_launch_num=40,
            duration=300,
            result_logging_dir="./real_llama70b/separate_offline/l4"
        )

    if model_name == "llama70b" and serving_mode == "offline" and method == "separate_t4":
        print("Running LLaMa 70B + Offline + Separate (T4)")
        os.makedirs("./real_llama70b/separate_offline/t4", exist_ok=True)
        run_heuristic_host_offline(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama70b/separate/real_sys_config_t4.txt",
            initial_launch_num=3,
            duration=300,
            result_logging_dir="./real_llama70b/separate_offline/t4"
        )

    if model_name == "llama70b" and serving_mode == "online" and method == "separate_a100":
        print("Running LLaMa 70B + Online + Separate (A100)")
        os.makedirs("./real_llama70b/separate_online/a100", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama70b/separate/real_sys_config_a100.txt",
            avg_throughput=240,
            duration=300,
            result_logging_dir="./real_llama70b/separate_online/a100"
        )

    if model_name == "llama70b" and serving_mode == "online" and method == "separate_l4":
        print("Running LLaMa 70B + Online + Separate (L4)")
        os.makedirs("./real_llama70b/separate_online/l4", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama70b/separate/real_sys_config_l4.txt",
            avg_throughput=120,
            duration=300,
            result_logging_dir="./real_llama70b/separate_online/l4"
        )

    if model_name == "llama70b" and serving_mode == "online" and method == "separate_t4":
        print("Running LLaMa 70B + Online + Separate (T4)")
        os.makedirs("./real_llama70b/separate_online/t4", exist_ok=True)
        run_heuristic_host_online(
            scheduler_name="random",
            real_sys_config_file_name="./layout_llama70b/separate/real_sys_config_t4.txt",
            avg_throughput=10,
            duration=300,
            result_logging_dir="./real_llama70b/separate_online/t4"
        )


if __name__ == '__main__':
    main()
