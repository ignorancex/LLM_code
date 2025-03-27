# 2024.10.29 Yixuan Mei
import sys
import os
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def petals_layout(model_name: ModelName, save_path: str):
    # heuristic method: petals
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster42.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=save_path,
        layout_method=LayoutMethod.Petals,
        machine_num_dict={"A100": 4, "V100": 6, "L4": 8, "L4x2": 4, "T4": 10, "T4x2": 6, "T4x4": 4}
    )
    petals_args = {
        "seed": 0,
        "max_out_links_per_node": 42,
    }
    layout_synthesizer.synthesize(args=petals_args)


def swarm_layout(model_name: ModelName, save_path: str, num_stages: int):
    # heuristic method: swarm
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster42.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=save_path,
        layout_method=LayoutMethod.Swarm,
        machine_num_dict={"A100": 4, "V100": 6, "L4": 8, "L4x2": 4, "T4": 10, "T4x2": 6, "T4x4": 4}
    )
    swarm_args = {
        "seed": 0,
        "num_stages": num_stages,  # as few as possible
        "max_out_links_per_node": 42,
    }
    layout_synthesizer.synthesize(args=swarm_args)


def ilp_layout70b(save_path: str):
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster42.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path=save_path,
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "V100": 6, "L4": 8, "L4x2": 4, "T4": 10, "T4x2": 6, "T4x4": 4}
    )

    # check if ./layout_llama70b/swarm/swarm_sol.ini exists
    if not os.path.exists("./layout_llama70b/swarm/swarm_sol.ini"):
        print("Please generate Swarm solution first, for this example Helix starts from the heuristic"
              "solution found by Swarm!")
        raise FileNotFoundError("File ./layout_llama70b/swarm/swarm_sol.ini not found!")

    # setting arguments for ILP layout synthesis
    ilp_args = {
        # pruning
        "enable_pruning": True,
        "min_keep": 12,
        "max_keep": 12,
        "keep_bandwidth_threshold": 1,
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
        "heuristic_sol_path": "./layout_llama70b/swarm/swarm_sol.ini",
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <ilp/swarm/petals/separate>"
    layout_method = sys.argv[1]
    assert layout_method in ["ilp", "swarm", "petals", "separate"], f"Invalid layout method {layout_method}."

    if layout_method == "ilp":
        # Helix's MILP-based model placement method
        os.makedirs("./layout_llama70b/ilp", exist_ok=True)
        ilp_layout70b("./layout_llama70b/ilp")
        print("Layout for LLaMa70B model is generated using ILP method.")

    elif layout_method == "swarm":
        # Heuristic method: swarm
        os.makedirs("./layout_llama70b/swarm", exist_ok=True)
        swarm_layout(ModelName.LLaMa70B, "./layout_llama70b/swarm", 20)
        print("Layout for LLaMa70B model is generated using Swarm method.")

    elif layout_method == "petals":
        # Heuristic method: petals
        os.makedirs("./layout_llama70b/petals", exist_ok=True)
        petals_layout(ModelName.LLaMa70B, "./layout_llama70b/petals")
        print("Layout for LLaMa70B model is generated using Petals method.")

    elif layout_method == "separate":
        print("We manually create model placement for the separate pipelines baseline.")
        print("The files are located in ./layout_llama70b/separate")

    else:
        raise ValueError(f"Invalid layout method: {layout_method}")


if __name__ == '__main__':
    main()
