# 2024.11.06 Yixuan Mei
import os.path
import sys
import pickle
from typing import Dict, List
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer
from simulator.event_simulator.cluster_simulator import ClusterSimulator, ModelName, SchedulingMethod, RequestPhase
from simulator.trace_generator.simulator_query_feeder import OnlineRequestFeeder, OfflineRequestFeeder
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import KVParameters, SchedulingMode


def get_latency(file_path: str) -> List[float]:
    with open(file_path, "rb") as f:
        latency_list = pickle.load(f)
    return [latency for arrival_time, latency in latency_list]


def analyze_latency(file_paths: List[str]) -> None:
    latency_list = []
    for file_path in file_paths:
        latency_list.extend(get_latency(file_path))
    latency_list.sort()
    print(f"Latency 5th percentile: {latency_list[int(len(latency_list) * 0.05)]:.2f} s")
    print(f"Latency 25th percentile: {latency_list[int(len(latency_list) * 0.25)]:.2f} s")
    print(f"Latency 50th percentile: {latency_list[int(len(latency_list) * 0.5)]:.2f} s")
    print(f"Latency 75th percentile: {latency_list[int(len(latency_list) * 0.75)]:.2f} s")
    print(f"Latency 95th percentile: {latency_list[int(len(latency_list) * 0.95)]:.2f} s")


def simulate_maxflow_online(
        model_name: ModelName,
        workspace_path: str,
        solution_file_name: str,
        complete_cluster_file_name: str,
        simulator_cluster_file_name: str,
        avg_throughput: float,
        machine_num_dict: Dict[str, int],
) -> float:
    # load cluster
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=complete_cluster_file_name,
        machine_profile_name="config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=workspace_path,
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
        "kv_param": KVParameters(expected_kv_hwm=0.9, expected_output_length_ratio=0.6),
        "scheduling_mode": SchedulingMode.Online,
    }
    simulator.init_scheduler(scheduling_method=SchedulingMethod.MaxFlow, args=scheduler_args)
    simulator.init_query_manager()
    simulator.mark_as_ready()

    # load the models into the simulator and update scheduler
    finish_model_loading_time = layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()

    # run simulation
    warm_up, duration = 0, 1800
    auto_test = OnlineRequestFeeder(cluster_token_throughput=avg_throughput,
                                    start_time=finish_model_loading_time,
                                    duration=duration, seed=0)
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
    # save prompt and decode latency
    prompt_latency_list, decode_latency_list = [], []
    # prompt and decode latency
    sum_prompt_latency, sum_decode_latency = 0, 0
    valid_prompts, valid_decodes = 0, 0
    for request_uid, time_request in simulator.finished_requests.items():
        time, request = time_request
        if analysis_start_time <= time <= analysis_end_time:
            if request.phase == RequestPhase.Initialization:
                sum_prompt_latency += request.location_history[-1][1] - request.location_history[0][1]
                prompt_latency_list.append((time, request.location_history[-1][1] - request.location_history[0][1]))
                valid_prompts += 1
            elif request.phase == RequestPhase.Increment:
                sum_decode_latency += request.location_history[-1][1] - request.location_history[0][1]
                decode_latency_list.append((time, request.location_history[-1][1] - request.location_history[0][1]))
                valid_decodes += 1
            else:
                assert False, "Found unknown requests phase!"

    # save the latency data with pickle
    prompt_file_name = os.path.join(workspace_path, "prompt_latency.pkl")
    decode_file_name = os.path.join(workspace_path, "decode_latency.pkl")
    with open(prompt_file_name, "wb") as f:
        pickle.dump(prompt_latency_list, f)
    with open(decode_file_name, "wb") as f:
        pickle.dump(decode_latency_list, f)
    return decode_throughput


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


def simulate_heuristic_online(
        model_name: ModelName,
        workspace_path: str,
        solution_file_name: str,
        complete_cluster_file_name: str,
        simulator_cluster_file_name: str,
        scheduling_method: SchedulingMethod,
        avg_throughput: float,
        machine_num_dict: Dict[str, int],
        force_set: bool = False,
) -> float:
    # load cluster
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=complete_cluster_file_name,
        machine_profile_name="config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=workspace_path,
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
    warm_up, duration = 0, 1800
    auto_test = OnlineRequestFeeder(cluster_token_throughput=avg_throughput,
                                    start_time=finish_model_loading_time,
                                    duration=duration, seed=0)
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
    # save prompt and decode latency
    prompt_latency_list, decode_latency_list = [], []
    # prompt and decode latency
    sum_prompt_latency, sum_decode_latency = 0, 0
    valid_prompts, valid_decodes = 0, 0
    for request_uid, time_request in simulator.finished_requests.items():
        time, request = time_request
        if analysis_start_time <= time <= analysis_end_time:
            if request.phase == RequestPhase.Initialization:
                sum_prompt_latency += request.location_history[-1][1] - request.location_history[0][1]
                prompt_latency_list.append((time, request.location_history[-1][1] - request.location_history[0][1]))
                valid_prompts += 1
            elif request.phase == RequestPhase.Increment:
                sum_decode_latency += request.location_history[-1][1] - request.location_history[0][1]
                decode_latency_list.append((time, request.location_history[-1][1] - request.location_history[0][1]))
                valid_decodes += 1
            else:
                assert False, "Found unknown requests phase!"

    # save the latency data with pickle
    prompt_file_name = os.path.join(workspace_path, "prompt_latency.pkl")
    decode_file_name = os.path.join(workspace_path, "decode_latency.pkl")
    with open(prompt_file_name, "wb") as f:
        pickle.dump(prompt_latency_list, f)
    with open(decode_file_name, "wb") as f:
        pickle.dump(decode_latency_list, f)
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

    return decode_throughput


def main():
    # parse arguments
    assert len(sys.argv) == 4, (f"Usage: python {sys.argv[0]} <helix/swarm/separate> <llama30b/llama70b>"
                                f" <online/offline>")
    method, model_name, serving_mode = sys.argv[1], sys.argv[2], sys.argv[3]
    assert method in ["helix", "swarm", "separate"], f"Invalid method: {method}"
    assert model_name in ["llama30b", "llama70b"], f"Invalid model name: {model_name}"
    assert serving_mode in ["online", "offline"], f"Invalid serving mode: {serving_mode}"

    # launch simulation
    if model_name == "llama30b":
        if serving_mode == "online":
            if method == "helix":
                os.makedirs("./simulation_llama30b/ilp_online/a100", exist_ok=True)
                os.makedirs("./simulation_llama30b/ilp_online/l4", exist_ok=True)
                os.makedirs("./simulation_llama30b/ilp_online/t4", exist_ok=True)
                a100_decode_throughput = simulate_maxflow_online(
                    model_name=ModelName.LLaMa30B,
                    workspace_path="./simulation_llama30b/ilp_online/a100",
                    solution_file_name="./layout_llama30b/ilp/a100/ilp_sol.ini",
                    complete_cluster_file_name="./config/a100.ini",
                    simulator_cluster_file_name="./layout_llama30b/ilp/a100/simulator_cluster.ini",
                    avg_throughput=600,
                    machine_num_dict={"A100": 4}
                )
                l4_decode_throughput = simulate_maxflow_online(
                    model_name=ModelName.LLaMa30B,
                    workspace_path="./simulation_llama30b/ilp_online/l4",
                    solution_file_name="./layout_llama30b/ilp/l4/ilp_sol.ini",
                    complete_cluster_file_name="./config/l4.ini",
                    simulator_cluster_file_name="./layout_llama30b/ilp/l4/simulator_cluster.ini",
                    avg_throughput=200,
                    machine_num_dict={"L4": 8}
                )
                t4_decode_throughput = simulate_maxflow_online(
                    model_name=ModelName.LLaMa30B,
                    workspace_path="./simulation_llama30b/ilp_online/t4",
                    solution_file_name="./layout_llama30b/ilp/t4/ilp_sol.ini",
                    complete_cluster_file_name="./config/t4.ini",
                    simulator_cluster_file_name="./layout_llama30b/ilp/t4/simulator_cluster.ini",
                    avg_throughput=170,
                    machine_num_dict={"T4": 12}
                )
                sum_decode_throughput = a100_decode_throughput + l4_decode_throughput + t4_decode_throughput
                print("*" * 60)
                print(f"LLaMa30B online simulation results: Helix")
                print(f"Total decode throughput: {sum_decode_throughput:.1f} tokens/s")
                print("Prompt latency:")
                analyze_latency(["./simulation_llama30b/ilp_online/a100/prompt_latency.pkl",
                                 "./simulation_llama30b/ilp_online/l4/prompt_latency.pkl",
                                 "./simulation_llama30b/ilp_online/t4/prompt_latency.pkl"])
                print("Decode latency:")
                analyze_latency(["./simulation_llama30b/ilp_online/a100/decode_latency.pkl",
                                 "./simulation_llama30b/ilp_online/l4/decode_latency.pkl",
                                 "./simulation_llama30b/ilp_online/t4/decode_latency.pkl"])
                print("*" * 60)

            elif method == "swarm":
                os.makedirs("./simulation_llama30b/swarm_online", exist_ok=True)
                decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa30B,
                    workspace_path="./simulation_llama30b/swarm_online",
                    solution_file_name="./layout_llama30b/swarm/swarm_sol.ini",
                    complete_cluster_file_name="./config/cluster24.ini",
                    simulator_cluster_file_name="./layout_llama30b/swarm/simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Swarm,
                    avg_throughput=450,
                    machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
                )
                print("*" * 60)
                print(f"LLaMa30B online simulation results: Swarm")
                print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
                print("Prompt latency:")
                analyze_latency(["./simulation_llama30b/swarm_online/prompt_latency.pkl"])
                print("Decode latency:")
                analyze_latency(["./simulation_llama30b/swarm_online/decode_latency.pkl"])
                print("*" * 60)

            elif method == "separate":
                os.makedirs("./simulation_llama30b/separate_online/a100", exist_ok=True)
                os.makedirs("./simulation_llama30b/separate_online/l4", exist_ok=True)
                os.makedirs("./simulation_llama30b/separate_online/t4", exist_ok=True)
                a100_decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa30B,
                    workspace_path="./simulation_llama30b/separate_online/a100",
                    solution_file_name="./layout_llama30b/separate/a100_solution_file.ini",
                    complete_cluster_file_name="./config/a100.ini",
                    simulator_cluster_file_name="./layout_llama30b/separate/a100_simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Naive,
                    avg_throughput=500,
                    machine_num_dict={"A100": 4}
                )
                l4_decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa30B,
                    workspace_path="./simulation_llama30b/separate_online/l4",
                    solution_file_name="./layout_llama30b/separate/l4_solution_file.ini",
                    complete_cluster_file_name="./config/l4.ini",
                    simulator_cluster_file_name="./layout_llama30b/separate/l4_simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Naive,
                    avg_throughput=150,
                    machine_num_dict={"L4": 8}
                )
                t4_decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa30B,
                    workspace_path="./simulation_llama30b/separate_online/t4",
                    solution_file_name="./layout_llama30b/separate/t4_solution_file.ini",
                    complete_cluster_file_name="./config/t4.ini",
                    simulator_cluster_file_name="./layout_llama30b/separate/t4_simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Naive,
                    avg_throughput=150,
                    machine_num_dict={"T4": 12}
                )
                sum_decode_throughput = a100_decode_throughput + l4_decode_throughput + t4_decode_throughput
                print("*" * 60)
                print(f"LLaMa30B online simulation results: Separate")
                print(f"Total decode throughput: {sum_decode_throughput:.1f} tokens/s")
                print("Prompt latency:")
                analyze_latency(["./simulation_llama30b/separate_online/a100/prompt_latency.pkl",
                                 "./simulation_llama30b/separate_online/l4/prompt_latency.pkl",
                                 "./simulation_llama30b/separate_online/t4/prompt_latency.pkl"])
                print("Decode latency:")
                analyze_latency(["./simulation_llama30b/separate_online/a100/decode_latency.pkl",
                                 "./simulation_llama30b/separate_online/l4/decode_latency.pkl",
                                 "./simulation_llama30b/separate_online/t4/decode_latency.pkl"])
                print("*" * 60)

            else:
                raise ValueError(f"Invalid method: {method}")

        elif serving_mode == "offline":
            if method == "helix":
                a100_decode_throughput = simulate_maxflow_offline(
                    model_name=ModelName.LLaMa30B,
                    solution_file_name="./layout_llama30b/ilp/a100/ilp_sol.ini",
                    complete_cluster_file_name="./config/a100.ini",
                    simulator_cluster_file_name="./layout_llama30b/ilp/a100/simulator_cluster.ini",
                    machine_num_dict={"A100": 4}
                )
                l4_decode_throughput = simulate_maxflow_offline(
                    model_name=ModelName.LLaMa30B,
                    solution_file_name="./layout_llama30b/ilp/l4/ilp_sol.ini",
                    complete_cluster_file_name="./config/l4.ini",
                    simulator_cluster_file_name="./layout_llama30b/ilp/l4/simulator_cluster.ini",
                    machine_num_dict={"L4": 8}
                )
                t4_decode_throughput = simulate_maxflow_offline(
                    model_name=ModelName.LLaMa30B,
                    solution_file_name="./layout_llama30b/ilp/t4/ilp_sol.ini",
                    complete_cluster_file_name="./config/t4.ini",
                    simulator_cluster_file_name="./layout_llama30b/ilp/t4/simulator_cluster.ini",
                    machine_num_dict={"T4": 12}
                )
                sum_decode_throughput = a100_decode_throughput + l4_decode_throughput + t4_decode_throughput
                print("*" * 60)
                print(f"LLaMa30B offline simulation results: Helix")
                print(f"Total decode throughput: {sum_decode_throughput:.1f} tokens/s")
                print("*" * 60)

            elif method == "swarm":
                decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa30B,
                    solution_file_name="./layout_llama30b/swarm/swarm_sol.ini",
                    complete_cluster_file_name="./config/cluster24.ini",
                    simulator_cluster_file_name="./layout_llama30b/swarm/simulator_cluster.ini",
                    initial_feed_num=50,
                    scheduling_method=SchedulingMethod.Swarm,
                    machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
                )
                print("*" * 60)
                print(f"LLaMa30B offline simulation results: Swarm")
                print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
                print("*" * 60)

            elif method == "separate":
                a100_decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa30B,
                    solution_file_name="./layout_llama30b/separate/a100_solution_file.ini",
                    complete_cluster_file_name="./config/a100.ini",
                    simulator_cluster_file_name="./layout_llama30b/separate/a100_simulator_cluster.ini",
                    initial_feed_num=30,
                    scheduling_method=SchedulingMethod.Naive,
                    machine_num_dict={"A100": 4}
                )
                l4_decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa30B,
                    solution_file_name="./layout_llama30b/separate/l4_solution_file.ini",
                    complete_cluster_file_name="./config/l4.ini",
                    simulator_cluster_file_name="./layout_llama30b/separate/l4_simulator_cluster.ini",
                    initial_feed_num=25,
                    scheduling_method=SchedulingMethod.Naive,
                    machine_num_dict={"L4": 8}
                )
                t4_decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa30B,
                    solution_file_name="./layout_llama30b/separate/t4_solution_file.ini",
                    complete_cluster_file_name="./config/t4.ini",
                    simulator_cluster_file_name="./layout_llama30b/separate/t4_simulator_cluster.ini",
                    initial_feed_num=25,
                    scheduling_method=SchedulingMethod.Naive,
                    machine_num_dict={"T4": 12}
                )
                sum_decode_throughput = a100_decode_throughput + l4_decode_throughput + t4_decode_throughput
                print("*" * 60)
                print(f"LLaMa30B offline simulation results: Separate")
                print(f"Total decode throughput: {sum_decode_throughput:.1f} tokens/s")
                print("*" * 60)

            else:
                raise ValueError(f"Invalid method: {method}")

        else:
            raise ValueError(f"Invalid serving mode: {serving_mode}")

    elif model_name == "llama70b":
        if serving_mode == "online":
            if method == "helix":
                os.makedirs("./simulation_llama70b/ilp_online", exist_ok=True)
                decode_throughput = simulate_maxflow_online(
                    model_name=ModelName.LLaMa70B,
                    workspace_path="./simulation_llama70b/ilp_online",
                    solution_file_name="./layout_llama70b/ilp/ilp_sol.ini",
                    complete_cluster_file_name="./config/cluster24.ini",
                    simulator_cluster_file_name="./layout_llama70b/ilp/simulator_cluster.ini",
                    avg_throughput=700,
                    machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
                )
                print("*" * 60)
                print(f"LLaMa70B online simulation results: Helix")
                print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
                print("Prompt latency:")
                analyze_latency(["./simulation_llama70b/ilp_online/prompt_latency.pkl"])
                print("Decode latency:")
                analyze_latency(["./simulation_llama70b/ilp_online/decode_latency.pkl"])
                print("*" * 60)

            elif method == "swarm":
                os.makedirs("./simulation_llama70b/swarm_online", exist_ok=True)
                decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa70B,
                    workspace_path="./simulation_llama70b/swarm_online",
                    solution_file_name="./layout_llama70b/swarm/swarm_sol.ini",
                    complete_cluster_file_name="./config/cluster24.ini",
                    simulator_cluster_file_name="./layout_llama70b/swarm/simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Swarm,
                    avg_throughput=350,
                    machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
                )
                print("*" * 60)
                print(f"LLaMa70B online simulation results: Swarm")
                print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
                print("Prompt latency:")
                analyze_latency(["./simulation_llama70b/swarm_online/prompt_latency.pkl"])
                print("Decode latency:")
                analyze_latency(["./simulation_llama70b/swarm_online/decode_latency.pkl"])
                print("*" * 60)

            elif method == "separate":
                os.makedirs("./simulation_llama70b/separate_online/a100", exist_ok=True)
                os.makedirs("./simulation_llama70b/separate_online/l4", exist_ok=True)
                os.makedirs("./simulation_llama70b/separate_online/t4", exist_ok=True)
                a100_decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa70B,
                    workspace_path="./simulation_llama70b/separate_online/a100",
                    solution_file_name="./layout_llama70b/separate/a100_solution_file.ini",
                    complete_cluster_file_name="./config/a100.ini",
                    simulator_cluster_file_name="./layout_llama70b/separate/a100_simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Naive,
                    avg_throughput=240,
                    machine_num_dict={"A100": 4},
                    force_set=True
                )
                l4_decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa70B,
                    workspace_path="./simulation_llama70b/separate_online/l4",
                    solution_file_name="./layout_llama70b/separate/l4_solution_file.ini",
                    complete_cluster_file_name="./config/l4.ini",
                    simulator_cluster_file_name="./layout_llama70b/separate/l4_simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Naive,
                    avg_throughput=120,
                    machine_num_dict={"L4": 8},
                    force_set=True
                )
                t4_decode_throughput = simulate_heuristic_online(
                    model_name=ModelName.LLaMa70B,
                    workspace_path="./simulation_llama70b/separate_online/t4",
                    solution_file_name="./layout_llama70b/separate/t4_solution_file.ini",
                    complete_cluster_file_name="./config/t4.ini",
                    simulator_cluster_file_name="./layout_llama70b/separate/t4_simulator_cluster.ini",
                    scheduling_method=SchedulingMethod.Naive,
                    avg_throughput=10,
                    machine_num_dict={"T4": 12},
                    force_set=True
                )
                sum_decode_throughput = a100_decode_throughput + l4_decode_throughput + t4_decode_throughput
                print("*" * 60)
                print(f"LLaMa70B online simulation results: Separate")
                print(f"Total decode throughput: {sum_decode_throughput:.1f} tokens/s")
                print("Prompt latency:")
                analyze_latency(["./simulation_llama70b/separate_online/a100/prompt_latency.pkl",
                                 "./simulation_llama70b/separate_online/l4/prompt_latency.pkl",
                                 "./simulation_llama70b/separate_online/t4/prompt_latency.pkl"])
                print("Decode latency:")
                analyze_latency(["./simulation_llama70b/separate_online/a100/decode_latency.pkl",
                                 "./simulation_llama70b/separate_online/l4/decode_latency.pkl",
                                 "./simulation_llama70b/separate_online/t4/decode_latency.pkl"])
                print("*" * 60)

            else:
                raise ValueError(f"Invalid method: {method}")

        elif serving_mode == "offline":
            if method == "helix":
                decode_throughput = simulate_maxflow_offline(
                    model_name=ModelName.LLaMa70B,
                    solution_file_name="./layout_llama70b/ilp/ilp_sol.ini",
                    complete_cluster_file_name="./config/cluster24.ini",
                    simulator_cluster_file_name="./layout_llama70b/ilp/simulator_cluster.ini",
                    machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
                )
                print("*" * 60)
                print(f"LLaMa70B offline simulation results: Helix")
                print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
                print("*" * 60)

            elif method == "swarm":
                decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa70B,
                    solution_file_name="./layout_llama70b/swarm/swarm_sol.ini",
                    complete_cluster_file_name="./config/cluster24.ini",
                    simulator_cluster_file_name="./layout_llama70b/swarm/simulator_cluster.ini",
                    initial_feed_num=180,
                    scheduling_method=SchedulingMethod.Swarm,
                    machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
                )
                print("*" * 60)
                print(f"LLaMa70B offline simulation results: Swarm")
                print(f"Total decode throughput: {decode_throughput:.1f} tokens/s")
                print("*" * 60)

            elif method == "separate":
                a100_decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa70B,
                    solution_file_name="./layout_llama70b/separate/a100_solution_file.ini",
                    complete_cluster_file_name="./config/a100.ini",
                    simulator_cluster_file_name="./layout_llama70b/separate/a100_simulator_cluster.ini",
                    initial_feed_num=18,
                    scheduling_method=SchedulingMethod.Naive,
                    machine_num_dict={"A100": 4},
                    force_set=True
                )
                l4_decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa70B,
                    solution_file_name="./layout_llama70b/separate/l4_solution_file.ini",
                    complete_cluster_file_name="./config/l4.ini",
                    simulator_cluster_file_name="./layout_llama70b/separate/l4_simulator_cluster.ini",
                    initial_feed_num=40,
                    scheduling_method=SchedulingMethod.Naive,
                    machine_num_dict={"L4": 8},
                    force_set=True
                )
                t4_decode_throughput = simulate_heuristic_offline(
                    model_name=ModelName.LLaMa70B,
                    solution_file_name="./layout_llama70b/separate/t4_solution_file.ini",
                    complete_cluster_file_name="./config/t4.ini",
                    simulator_cluster_file_name="./layout_llama70b/separate/t4_simulator_cluster.ini",
                    initial_feed_num=3,
                    scheduling_method=SchedulingMethod.Naive,
                    machine_num_dict={"T4": 12},
                    force_set=True
                )
                sum_decode_throughput = a100_decode_throughput + l4_decode_throughput + t4_decode_throughput
                print("*" * 60)
                print(f"LLaMa70B offline simulation results: Separate")
                print(f"Total decode throughput: {sum_decode_throughput:.1f} tokens/s")
                print("*" * 60)

            else:
                raise ValueError(f"Invalid method: {method}")

        else:
            raise ValueError(f"Invalid serving mode: {serving_mode}")

    else:
        raise ValueError(f"Invalid model name: {model_name}")


if __name__ == '__main__':
    main()
