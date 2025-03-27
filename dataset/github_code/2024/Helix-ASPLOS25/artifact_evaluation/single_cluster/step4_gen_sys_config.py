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
        machine_profile_file_name="./config/machine_profiles.ini",
        # model placement
        solution_file_name=solution_file_name,
        simulator_cluster_file_name=simulator_cluster_file_name,
        # output directory
        output_dir=output_dir,
        output_file_name="real_sys_config.txt"
    )


def main():
    # parse arguments
    assert len(sys.argv) == 3, f"Usage: python {sys.argv[0]} <helix/swarm/separate> <llama30b/llama70b>"
    method, model_name = sys.argv[1], sys.argv[2]
    assert method in ["helix", "swarm", "separate"], f"Invalid method: {method}"
    assert model_name in ["llama30b", "llama70b"], f"Invalid model name: {model_name}"

    # Generate the real system configuration
    if model_name == "llama30b" and method == "helix":
        os.makedirs("./layout_llama30b/ilp/a100", exist_ok=True)
        os.makedirs("./layout_llama30b/ilp/l4", exist_ok=True)
        os.makedirs("./layout_llama30b/ilp/t4", exist_ok=True)
        # sub-cluster of a100
        generate_real_system_config(
            model_name=ModelName.LLaMa30B,
            complete_cluster_file_name="./config/a100.ini",
            solution_file_name="./layout_llama30b/ilp/a100/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/a100/simulator_cluster.ini",
            output_dir="./layout_llama30b/ilp/a100",
            machine_num_dict={"A100": 4}
        )
        # sub-cluster of l4
        generate_real_system_config(
            model_name=ModelName.LLaMa30B,
            complete_cluster_file_name="./config/l4.ini",
            solution_file_name="./layout_llama30b/ilp/l4/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/l4/simulator_cluster.ini",
            output_dir="./layout_llama30b/ilp/l4",
            machine_num_dict={"L4": 8}
        )
        # sub-cluster of t4
        generate_real_system_config(
            model_name=ModelName.LLaMa30B,
            complete_cluster_file_name="./config/t4.ini",
            solution_file_name="./layout_llama30b/ilp/t4/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/ilp/t4/simulator_cluster.ini",
            output_dir="./layout_llama30b/ilp/t4",
            machine_num_dict={"T4": 12}
        )
        print("Generated real system configurations for LLaMa30B with Helix")

    if model_name == "llama30b" and method == "swarm":
        os.makedirs("./layout_llama30b/swarm", exist_ok=True)
        generate_real_system_config(
            model_name=ModelName.LLaMa30B,
            complete_cluster_file_name="./config/cluster24.ini",
            solution_file_name="./layout_llama30b/swarm/swarm_sol.ini",
            simulator_cluster_file_name="./layout_llama30b/swarm/simulator_cluster.ini",
            output_dir="./layout_llama30b/swarm",
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )

    if model_name == "llama30b" and method == "separate":
        os.makedirs("./layout_llama30b/separate/a100", exist_ok=True)
        os.makedirs("./layout_llama30b/separate/l4", exist_ok=True)
        os.makedirs("./layout_llama30b/separate/t4", exist_ok=True)
        # sub-cluster of a100
        generate_real_system_config(
            model_name=ModelName.LLaMa30B,
            complete_cluster_file_name="./config/a100.ini",
            solution_file_name="./layout_llama30b/separate/a100_solution_file.ini",
            simulator_cluster_file_name="./layout_llama30b/separate/a100_simulator_cluster.ini",
            output_dir="./layout_llama30b/separate/a100",
            machine_num_dict={"A100": 4}
        )
        # sub-cluster of l4
        generate_real_system_config(
            model_name=ModelName.LLaMa30B,
            complete_cluster_file_name="./config/l4.ini",
            solution_file_name="./layout_llama30b/separate/l4_solution_file.ini",
            simulator_cluster_file_name="./layout_llama30b/separate/l4_simulator_cluster.ini",
            output_dir="./layout_llama30b/separate/l4",
            machine_num_dict={"L4": 8}
        )
        # sub-cluster of t4
        generate_real_system_config(
            model_name=ModelName.LLaMa30B,
            complete_cluster_file_name="./config/t4.ini",
            solution_file_name="./layout_llama30b/separate/t4_solution_file.ini",
            simulator_cluster_file_name="./layout_llama30b/separate/t4_simulator_cluster.ini",
            output_dir="./layout_llama30b/separate/t4",
            machine_num_dict={"T4": 12}
        )

    if model_name == "llama70b" and method == "helix":
        os.makedirs("./layout_llama70b/ilp", exist_ok=True)
        generate_real_system_config(
            model_name=ModelName.LLaMa70B,
            complete_cluster_file_name="./config/cluster24.ini",
            solution_file_name="./layout_llama70b/ilp/ilp_sol.ini",
            simulator_cluster_file_name="./layout_llama70b/ilp/simulator_cluster.ini",
            output_dir="./layout_llama70b/ilp",
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )

    if model_name == "llama70b" and method == "swarm":
        os.makedirs("./layout_llama70b/swarm", exist_ok=True)
        generate_real_system_config(
            model_name=ModelName.LLaMa70B,
            complete_cluster_file_name="./config/cluster24.ini",
            solution_file_name="./layout_llama70b/swarm/swarm_sol.ini",
            simulator_cluster_file_name="./layout_llama70b/swarm/simulator_cluster.ini",
            output_dir="./layout_llama70b/swarm",
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )

    if model_name == "llama70b" and method == "separate":
        print("We manually generated it in: ./layout_llama70b/separate")


if __name__ == '__main__':
    main()
