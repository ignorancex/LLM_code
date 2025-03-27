# 2024.11.06 Yixuan Mei
import os.path
import sys
import pickle
from typing import Dict, List
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer
from simulator.event_simulator.cluster_simulator import ClusterSimulator, ModelName, SchedulingMethod, RequestPhase
from simulator.trace_generator.simulator_query_feeder import OnlineRequestFeeder, OfflineRequestFeeder
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import KVParameters, SchedulingMode

def simulate_maxflow_offline(
        model_name: ModelName,
        solution_file_name: str,
        complete_cluster_file_name: str,
        simulator_cluster_file_name: str,
        machine_num_dict: Dict[str, int],
):
    # load cluster
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=complete_cluster_file_name,
        machine_profile_name="config_distributed/machine_profiles.ini",
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
    auto_test.auto_simulate(simulator=simulator, watch_items=["all"], watch_interval=200)

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


def simulate_heuristic_offline(
        model_name: ModelName,
        solution_file_name: str,
        complete_cluster_file_name: str,
        simulator_cluster_file_name: str,
        initial_feed_num: int,
        scheduling_method: SchedulingMethod,
        machine_num_dict: Dict[str, int],
        force_set: bool = False,
) -> float:
    # load cluster
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=complete_cluster_file_name,
        machine_profile_name="config_distributed/machine_profiles.ini",
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
    simulator.model_manager.allow_force_set = force_set  # for baseline: separate pipelines
    simulator.from_ini_file(config_file_name=cluster_file_path)
    simulator.init_scheduler(scheduling_method=scheduling_method, args=None)
    simulator.init_query_manager()
    simulator.mark_as_ready()

    # load the models into the simulator and update scheduler
    finish_model_loading_time = layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()

    # run simulation
    warm_up, duration = 60, 600
    auto_test = OfflineRequestFeeder(initial_query_count=initial_feed_num, start_time=finish_model_loading_time,
                                     duration=warm_up + duration, stop_at_duration=True, feed_hwm=0.8, seed=0)
    auto_test.auto_simulate(simulator=simulator, watch_items=["all"], watch_interval=200)

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

    # analyze the congestion
    print("*" * 60)
    print(f"Congestion analysis: {scheduling_method}")
    location2sum_time, location2count = {}, {}
    for request_uid, time_request in simulator.finished_requests.items():
        time, request = time_request
        if not (analysis_start_time <= time <= analysis_end_time):
            continue
        if request.phase == RequestPhase.Initialization:
            # initialization is a better sign of congestion
            for idx, history_entry in enumerate(request.location_history):
                location, arrival_time = history_entry
                location: str
                arrival_time: int
                if "ComputeNode" in location:
                    _, leave_time = request.location_history[idx + 1]
                    delta_time = leave_time - arrival_time
                    if location not in location2sum_time:
                        location2sum_time[location] = delta_time
                        location2count[location] = 1
                    else:
                        location2sum_time[location] += delta_time
                        location2count[location] += 1
    for location, sum_time in location2sum_time.items():
        print(f"{location}: {sum_time / location2count[location]}s")
    print("*" * 60)

    return decode_throughput


def main():
    # parse arguments
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <helix/swarm/random/shortest_queue>"
    method = sys.argv[1]
    assert method in ["helix", "swarm", "random", "shortest_queue"], "Unknown method!"

    # launch simulation
    if method == "helix":
        decode_throughput = simulate_maxflow_offline(
            model_name=ModelName.LLaMa70B,
            solution_file_name="./layout_distributed/ilp_sol.ini",
            complete_cluster_file_name="./config_distributed/3cluster24.ini",
            simulator_cluster_file_name="./layout_distributed/simulator_cluster.ini",
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )
        print("*" * 60)
        print(f"LLaMa70B offline simulation results: Helix")
        print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
        print("*" * 60)

    elif method == "swarm":
        decode_throughput = simulate_heuristic_offline(
            model_name=ModelName.LLaMa70B,
            solution_file_name="./layout_distributed/ilp_sol.ini",
            complete_cluster_file_name="./config_distributed/3cluster24.ini",
            simulator_cluster_file_name="./layout_distributed/simulator_cluster.ini",
            initial_feed_num=180,
            scheduling_method=SchedulingMethod.Swarm,
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )
        print("*" * 60)
        print(f"LLaMa70B offline simulation results: Swarm")
        print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
        print("*" * 60)

    elif method == "random":
        decode_throughput = simulate_heuristic_offline(
            model_name=ModelName.LLaMa70B,
            solution_file_name="./layout_distributed/ilp_sol.ini",
            complete_cluster_file_name="./config_distributed/3cluster24.ini",
            simulator_cluster_file_name="./layout_distributed/simulator_cluster.ini",
            initial_feed_num=180,
            scheduling_method=SchedulingMethod.Naive,
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )
        print("*" * 60)
        print(f"LLaMa70B offline simulation results: Random")
        print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
        print("*" * 60)

    elif method == "shortest_queue":
        decode_throughput = simulate_heuristic_offline(
            model_name=ModelName.LLaMa70B,
            solution_file_name="./layout_distributed/ilp_sol.ini",
            complete_cluster_file_name="./config_distributed/3cluster24.ini",
            simulator_cluster_file_name="./layout_distributed/simulator_cluster.ini",
            initial_feed_num=180,
            scheduling_method=SchedulingMethod.ShortestQueue,
            machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
        )
        print("*" * 60)
        print(f"LLaMa70B offline simulation results: ShortestQueue")
        print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
        print("*" * 60)

    else:
        assert False, "Unknown method!"



if __name__ == '__main__':
    main()
