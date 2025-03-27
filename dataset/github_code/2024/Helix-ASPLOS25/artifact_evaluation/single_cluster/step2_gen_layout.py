# 2024.10.29 Yixuan Mei
import sys
import os
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def petals_layout(model_name: ModelName, save_path: str):
    # heuristic method: petals
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster24.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=save_path,
        layout_method=LayoutMethod.Petals,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    petals_args = {
        "seed": 0,
        "max_out_links_per_node": 24,
    }
    layout_synthesizer.synthesize(args=petals_args)


def swarm_layout(model_name: ModelName, save_path: str, num_stages: int):
    # heuristic method: swarm
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster24.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=save_path,
        layout_method=LayoutMethod.Swarm,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    swarm_args = {
        "seed": 0,
        "num_stages": num_stages,  # as few as possible
        "max_out_links_per_node": 24,
    }
    layout_synthesizer.synthesize(args=swarm_args)


def ilp_layout30b(cluster_file_path: str, save_path: str, machine_num_dict: dict, heuristic_path: str):
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=cluster_file_path,
        machine_profile_name="./config/machine_profiles.ini",
        model_name=ModelName.LLaMa30B,
        workspace_path=save_path,
        layout_method=LayoutMethod.ILP,
        machine_num_dict=machine_num_dict
    )

    ilp_args = {
        "seed": 1,
        # pruning
        "enable_pruning": False,
        "min_keep": 12,
        "max_keep": 12,
        "keep_bandwidth_threshold": 1 * mbps,
        # ILP
        "use_existing_sol": False,
        "allow_partial_inference": False,
        "remove_redundant": False,
        "max_run_time": 1,
        "early_stop_time": 100,
        "early_stop_threshold": 0.9,
        "existing_sol_path": "path/to/existing/ilp_solution.sol",
        # heuristic
        "start_from_heuristic": True,
        "heuristic_sol_path": heuristic_path,
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def ilp_layout70b(save_path: str):
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster24.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path=save_path,
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )

    # check if ./layout_llama70b/petals/petals_sol.ini exists
    if not os.path.exists("./layout_llama70b/petals/petals_sol.ini"):
        print("Please generate Petals solution first, for this example Helix starts from the heuristic"
              "solution found by Petals!")
        raise FileNotFoundError("File ./layout_llama70b/petals/petals_sol.ini not found!")

    # setting arguments for ILP layout synthesis
    ilp_args = {
        # pruning
        "enable_pruning": False,
        "min_keep": 12,
        "max_keep": 12,
        "keep_bandwidth_threshold": 1 * mbps,
        # ILP
        "use_existing_sol": False,
        "allow_partial_inference": False,
        "remove_redundant": True,
        "max_run_time": 36000,
        "early_stop_time": 100,
        "early_stop_threshold": 0.95,
        "existing_sol_path": "path/to/existing/ilp_solution.sol",
        # heuristic
        "start_from_heuristic": True,
        "heuristic_sol_path": "./layout_llama70b/petals/petals_sol.ini",
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    assert len(sys.argv) == 3, f"Usage: python {sys.argv[0]} <ilp/swarm/petals/separate> <llama30b/llama70b>"
    layout_method = sys.argv[1]
    model_name = sys.argv[2]

    if layout_method == "ilp":
        # Helix's MILP-based model placement method
        if model_name == "llama30b":
            os.makedirs("./layout_llama30b/ilp/a100", exist_ok=True)
            os.makedirs("./layout_llama30b/ilp/l4", exist_ok=True)
            os.makedirs("./layout_llama30b/ilp/t4", exist_ok=True)
            ilp_layout30b(
                cluster_file_path="./config/a100.ini",
                save_path="./layout_llama30b/ilp/a100",
                machine_num_dict={"A100": 4},
                heuristic_path="./layout_llama30b/separate/a100_solution_file.ini"
            )
            ilp_layout30b(
                cluster_file_path="./config/l4.ini",
                save_path="./layout_llama30b/ilp/l4",
                machine_num_dict={"L4": 8},
                heuristic_path="./layout_llama30b/separate/l4_solution_file.ini"
            )
            ilp_layout30b(
                cluster_file_path="./config/t4.ini",
                save_path="./layout_llama30b/ilp/t4",
                machine_num_dict={"T4": 12},
                heuristic_path="./layout_llama30b/separate/t4_solution_file.ini"
            )
            print("Layout for LLaMa30B model is generated using ILP method.")
        elif model_name == "llama70b":
            os.makedirs("./layout_llama70b/ilp", exist_ok=True)
            ilp_layout70b("./layout_llama70b/ilp")
            print("Layout for LLaMa70B model is generated using ILP method.")

        else:
            raise ValueError(f"Invalid model name: {model_name}")

    elif layout_method == "swarm":
        # Heuristic method: swarm
        if model_name == "llama30b":
            os.makedirs("./layout_llama30b/swarm", exist_ok=True)
            swarm_layout(ModelName.LLaMa30B, "./layout_llama30b/swarm", 10)
            print("Layout for LLaMa30B model is generated using Swarm method.")

        elif model_name == "llama70b":
            os.makedirs("./layout_llama70b/swarm", exist_ok=True)
            swarm_layout(ModelName.LLaMa70B, "./layout_llama70b/swarm", 20)
            print("Layout for LLaMa70B model is generated using Swarm method.")

        else:
            raise ValueError(f"Invalid model name: {model_name}")

    elif layout_method == "petals":
        # Heuristic method: petals
        if model_name == "llama30b":
            os.makedirs("./layout_llama30b/petals", exist_ok=True)
            petals_layout(ModelName.LLaMa30B, "./layout_llama30b/petals")
            print("Layout for LLaMa30B model is generated using Petals method.")

        elif model_name == "llama70b":
            os.makedirs("./layout_llama70b/petals", exist_ok=True)
            petals_layout(ModelName.LLaMa70B, "./layout_llama70b/petals")
            print("Layout for LLaMa70B model is generated using Petals method.")

        else:
            raise ValueError(f"Invalid model name: {model_name}")

    elif layout_method == "separate":
        print("We manually create model placement for the separate pipelines baseline.")
        if model_name == "llama30b":
            print("The files are located in ./layout_llama30b/separate")

        elif model_name == "llama70b":
            print("The files are located in ./layout_llama70b/separate")

        else:
            raise ValueError(f"Invalid model name: {model_name}")

    else:
        raise ValueError(f"Invalid layout method: {layout_method}")


if __name__ == '__main__':
    main()
