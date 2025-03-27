from typing import Dict
from simulator.event_simulator.cluster_simulator import ModelName
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName


def ilp_synthesis(
        # model and machine
        machine_num_dict: Dict[str, int],
        machine_profile_name: str,
        model_name: ModelName,
        # cluster
        complete_cluster_file_name: str,
        # running settings
        ilp_args,
        workspace_path: str
) -> None:
    layout_synthesizer = LayoutSynthesizer(complete_cluster_file_name=complete_cluster_file_name,
                                           machine_profile_name=machine_profile_name,
                                           model_name=model_name,
                                           workspace_path=workspace_path,
                                           layout_method=LayoutMethod.ILP,
                                           machine_num_dict=machine_num_dict)
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    ilp_synthesis(
        machine_num_dict={"L4": 4, "T4": 6},
        machine_profile_name="./config/machine_profiles.ini",
        model_name=ModelName.LLaMa30B,
        complete_cluster_file_name="./config/cluster10.ini",
        ilp_args={
            # pruning
            "enable_pruning": False,
            "min_keep": 10,
            "max_keep": 10,
            "keep_bandwidth_threshold": 1,
            # ILP
            "use_existing_sol": False,
            "allow_partial_inference": False,
            "remove_redundant": True,
            "max_run_time": 360000,
            "early_stop_time": 100,
            "early_stop_threshold": 0.95,
            "existing_sol_path": None,
            # heuristic
            "start_from_heuristic": False,
            "heuristic_sol_path": None,
        },
        workspace_path="./layouts/quality"
    )


if __name__ == '__main__':
    main()
