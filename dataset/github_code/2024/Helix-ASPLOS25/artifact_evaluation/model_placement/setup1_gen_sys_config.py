import os
import sys
from llm_sys.gen_sys_config import gen_sys_config
from simulator.event_simulator.cluster_simulator import ModelName


def generate_real_system_config(
        model_name: ModelName,
        complete_cluster_file_name: str,  # e.g.: "./config/single24.ini"
        solution_file_name: str,  # e.g.: "./layout/ilp_sol.ini"
        simulator_cluster_file_name: str,  # e.g.: "./layout/simulator_cluster.ini"
        output_dir: str,
        machine_num_dict,
):
    type2ips = {}
    if "A100" in machine_num_dict:
        type2ips["A100"] = ["10.128.0.14", "10.128.0.41", "10.128.0.42", "10.128.0.43"]
    if "L4" in machine_num_dict:
        type2ips["L4"] = ["10.128.0.16", "10.128.0.20", "10.128.0.21", "10.128.0.22",
                          "10.128.0.23", "10.128.0.24", "10.128.0.25", "10.128.0.30"]
    if "T4" in machine_num_dict:
        type2ips["T4"] = ["10.128.0.15", "10.128.0.26", "10.128.0.27", "10.128.0.28",
                          "10.128.0.32", "10.128.0.33", "10.128.0.34", "10.128.0.35",
                          "10.128.0.36", "10.128.0.37", "10.128.0.40", "10.128.0.39"]
    # Generate the system configuration
    gen_sys_config(
        host_ip="10.128.0.31",
        type2ips=type2ips,
        # model and machine
        machine_num_dict=machine_num_dict,
        model_name=model_name,
        # cluster
        complete_cluster_file_name=complete_cluster_file_name,
        machine_profile_file_name="./config_single/machine_profiles.ini",
        # model placement
        solution_file_name=solution_file_name,
        simulator_cluster_file_name=simulator_cluster_file_name,
        # output directory
        output_dir=output_dir,
        output_file_name="real_sys_config.txt"
    )


def main():
    # parse arguments
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <helix/swarm/petals>"
    method = sys.argv[1]
    assert method in ["helix", "swarm", "petals"], f"Invalid method: {method}"

    # Generate the real system configuration
    if method == "helix":
        os.makedirs("./layout_single/ilp", exist_ok=True)
        generate_real_system_config(
            model_name=ModelName.LLaMa70B,
            complete_cluster_file_name="./config_single/cluster24.ini",
            solution_file_name="./layout_single/ilp/ilp_sol.ini",
            simulator_cluster_file_name="./layout_single/ilp/simulator_cluster.ini",
            output_dir="./layout_single/ilp",
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )

    if method == "swarm":
        os.makedirs("./layout_single/swarm", exist_ok=True)
        generate_real_system_config(
            model_name=ModelName.LLaMa70B,
            complete_cluster_file_name="./config_single/cluster24.ini",
            solution_file_name="./layout_single/swarm/swarm_sol.ini",
            simulator_cluster_file_name="./layout_single/swarm/simulator_cluster.ini",
            output_dir="./layout_single/swarm",
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )

    if method == "petals":
        os.makedirs("./layout_single/petals", exist_ok=True)
        generate_real_system_config(
            model_name=ModelName.LLaMa70B,
            complete_cluster_file_name="./config_single/cluster24.ini",
            solution_file_name="./layout_single/petals/petals_sol.ini",
            simulator_cluster_file_name="./layout_single/petals/simulator_cluster.ini",
            output_dir="./layout_single/petals",
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )


if __name__ == '__main__':
    main()
