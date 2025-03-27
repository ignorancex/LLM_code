# 2024.11.07 Yixuan Mei

import sys
import os
from typing import Dict, List
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName
from simulator.event_simulator.cluster_simulator import ClusterSimulator, SchedulingMethod, RequestPhase
from simulator.trace_generator.simulator_query_feeder import OnlineRequestFeeder, OfflineRequestFeeder
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import KVParameters, SchedulingMode


def cluster24_ilp_layout70b():
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/3cluster24.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/raw_24",
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )

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
        "start_from_heuristic": False,
        "heuristic_sol_path": "",
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def cluster42_ilp_layout70b():
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster42.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/raw_42",
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "V100": 6, "L4": 8, "L4x2": 4, "T4": 10, "T4x2": 6, "T4x4": 4}
    )

    # setting arguments for ILP layout synthesis
    ilp_args = {
        # pruning
        "enable_pruning": True,
        "min_keep": 12,
        "max_keep": 12,
        "keep_bandwidth_threshold": -1,
        # ILP
        "use_existing_sol": False,
        "allow_partial_inference": False,
        "remove_redundant": True,
        "max_run_time": 36000,
        "early_stop_time": 100,
        "early_stop_threshold": 0.95,
        "existing_sol_path": "path/to/existing/ilp_solution.sol",
        # heuristic
        "start_from_heuristic": False,
        "heuristic_sol_path": "",
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <24/42>"
    setup_name = sys.argv[1]
    assert setup_name in ["24", "42"], f"Invalid setup name: {setup_name}"

    # run
    if setup_name == "24":
        cluster24_ilp_layout70b()
        print("Layout synthesis for 24-node cluster completed (No pruning)")

    elif setup_name == "42":
        cluster42_ilp_layout70b()
        print("Layout synthesis for 42-node cluster completed (No pruning)")

    else:
        raise ValueError(f"Invalid setup name: {setup_name}")


if __name__ == '__main__':
    main()
