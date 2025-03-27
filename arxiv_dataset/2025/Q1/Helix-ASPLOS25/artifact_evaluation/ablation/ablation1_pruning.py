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
        workspace_path="./layouts/no_prune_24",
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )

    # setting arguments for ILP layout synthesis
    ilp_args = {
        # pruning
        "enable_pruning": False,
        "min_keep": -1,
        "max_keep": -1,
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
        "start_from_heuristic": True,
        "heuristic_sol_path": "./heuristics/cluster24/swarm_sol.ini",
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def cluster42_ilp_layout70b():
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster42.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/no_prune_42",
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "V100": 6, "L4": 8, "L4x2": 4, "T4": 10, "T4x2": 6, "T4x4": 4}
    )

    # setting arguments for ILP layout synthesis
    ilp_args = {
        # pruning
        "enable_pruning": False,
        "min_keep": -1,
        "max_keep": -1,
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
        "start_from_heuristic": True,
        "heuristic_sol_path": "./heuristics/cluster42/swarm_sol.ini",
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def simulate_maxflow_offline(
        solution_file_name: str,
        complete_cluster_file_name: str,
        simulator_cluster_file_name: str,
        machine_num_dict: Dict[str, int],
):
    # load cluster
    model_name = ModelName.LLaMa70B
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=complete_cluster_file_name,
        machine_profile_name="config/machine_profiles.ini",
        model_name=model_name,
        workspace_path="./tmp",
        layout_method=LayoutMethod.LoadExisting,
        machine_num_dict=machine_num_dict
    )
    layout_args = {
        "solution_file_name": solution_file_name,
        "simulator_cluster_file_name": simulator_cluster_file_name,
    }
    cluster_file_path = layout_synthesizer.synthesize(args=layout_args)

    # initialize the simulator
    simulator = ClusterSimulator(model_name=model_name, machine_num_dict=machine_num_dict)
    simulator.from_ini_file(config_file_name=cluster_file_path)
    scheduler_args = {
        # offline
        "kv_param": KVParameters(expected_kv_hwm=0.85, expected_output_length_ratio=1),
        "scheduling_mode": SchedulingMode.Offline,
    }
    simulator.init_scheduler(scheduling_method=SchedulingMethod.MaxFlow, args=scheduler_args)
    simulator.init_query_manager()
    simulator.mark_as_ready()

    # load the models into the simulator and update scheduler
    finish_model_loading_time = layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()

    # run simulation
    warm_up, duration = 60, 600
    auto_test = OfflineRequestFeeder(initial_query_count=20, start_time=finish_model_loading_time,
                                     duration=warm_up + duration, stop_at_duration=True, feed_hwm=0.8, seed=0)
    auto_test.auto_simulate(simulator=simulator, watch_items=["all"], watch_interval=10)

    # result processing
    analysis_start_time = finish_model_loading_time + warm_up
    analysis_end_time = finish_model_loading_time + warm_up + duration
    # decode throughput
    total_tokens = 0
    for request_uid, time_request in simulator.finished_requests.items():
        time, request = time_request
        if request.phase == RequestPhase.Initialization:
            continue
        if analysis_start_time <= time <= analysis_end_time:
            assert request.token_seq_length == 1, "Only count decode requests!"
            total_tokens += request.token_seq_length
    decode_throughput = total_tokens / duration
    # prompt and decode latency
    sum_prompt_latency, sum_decode_latency = 0, 0
    valid_prompts, valid_decodes = 0, 0
    for request_uid, time_request in simulator.finished_requests.items():
        time, request = time_request
        if analysis_start_time <= time <= analysis_end_time:
            if request.phase == RequestPhase.Initialization:
                sum_prompt_latency += request.location_history[-1][1] - request.location_history[0][1]
                valid_prompts += 1
            elif request.phase == RequestPhase.Increment:
                sum_decode_latency += request.location_history[-1][1] - request.location_history[0][1]
                valid_decodes += 1
            else:
                assert False, "Found unknown requests phase!"

    return decode_throughput


def main():
    assert len(sys.argv) == 3, f"Usage: python {sys.argv[0]} <layout/simulate> <24/42>"
    mode, setup_name = sys.argv[1], sys.argv[2]
    assert mode in ["layout", "simulate"], f"Invalid mode: {mode}"
    assert setup_name in ["24", "42"], f"Invalid setup name: {setup_name}"

    # run
    if mode == "layout":
        if setup_name == "24":
            cluster24_ilp_layout70b()
            print("Layout synthesis for 24-node cluster completed (No pruning)")

        elif setup_name == "42":
            cluster42_ilp_layout70b()
            print("Layout synthesis for 42-node cluster completed (No pruning)")

        else:
            raise ValueError(f"Invalid setup name: {setup_name}")

    elif mode == "simulate":
        if setup_name == "24":
            decode_throughput = simulate_maxflow_offline(
                solution_file_name="./layouts/no_prune_24/ilp_sol.ini",
                complete_cluster_file_name="./config/3cluster24.ini",
                simulator_cluster_file_name="./layouts/no_prune_24/simulator_cluster.ini",
                machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
            )
            print("*" * 60)
            print(f"Decode throughput for 24-node cluster: {decode_throughput} (w/o pruning)")
            print("*" * 60)

        elif setup_name == "42":
            decode_throughput = simulate_maxflow_offline(
                solution_file_name="./layouts/no_prune_42/ilp_sol.ini",
                complete_cluster_file_name="./config/cluster42.ini",
                simulator_cluster_file_name="./layouts/no_prune_42/simulator_cluster.ini",
                machine_num_dict={"A100": 4, "V100": 6, "L4": 8, "L4x2": 4, "T4": 10, "T4x2": 6, "T4x4": 4}
            )
            print("*" * 60)
            print(f"Decode throughput for 42-node cluster: {decode_throughput} (w/o pruning)")
            print("*" * 60)

        else:
            raise ValueError(f"Invalid setup name: {setup_name}")

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == '__main__':
    main()
