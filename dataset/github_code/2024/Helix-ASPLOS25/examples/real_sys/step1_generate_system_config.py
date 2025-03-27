from llm_sys.gen_sys_config import gen_sys_config
from simulator.event_simulator.cluster_simulator import ModelName


def main():
    gen_sys_config(
        host_ip="10.128.0.31",
        type2ips={"A100": ["10.128.0.14", "10.128.0.17", "10.128.0.18", "10.128.0.19"],
                  "L4": ["10.128.0.16", "10.128.0.20", "10.128.0.21", "10.128.0.22",
                         "10.128.0.23", "10.128.0.24", "10.128.0.25", "10.128.0.30"],
                  "T4": ["10.128.0.15", "10.128.0.26", "10.128.0.27", "10.128.0.28",
                         "10.128.0.32", "10.128.0.33", "10.128.0.34", "10.128.0.35",
                         "10.128.0.36", "10.128.0.37", "10.128.0.40", "10.128.0.39"]},
        # model and machine
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
        model_name=ModelName.LLaMa70B,
        # cluster
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_file_name="./config/machine_profile.ini",
        # model placement
        solution_file_name="./layout/ilp_sol.ini",
        simulator_cluster_file_name="./layout/simulator_cluster.ini",
        # output directory
        output_dir="./config",
        output_file_name="real_sys_config.txt"
    )
    print("System config generated to ./config/real_sys_config.txt!")


if __name__ == '__main__':
    main()
