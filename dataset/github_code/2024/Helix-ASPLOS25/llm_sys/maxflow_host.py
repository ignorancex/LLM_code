# 2024.04.24 Yixuan Mei
import os.path
import time

import llm_host
from typing import Dict, List, Tuple

from simulator.event_simulator.cluster_simulator import ClusterSimulator, ModelName, SchedulingMethod
from simulator.event_simulator.request import InferenceRequest, RequestPhase, PipelineStage
from simulator.initial_layout.layout_synthesizer import LayoutSynthesizer, LayoutMethod
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import SchedulingMode, KVParameters, SchedulerCore
from simulator.trace_generator.trace_generator import TraceGenerator, LengthSampler, ArrivalRateSource, Dataset

from llm_sys.utils import SIMULATOR_NODE_OFFSET, get_local_ip, CONFIG_BROADCAST_ADDR, FlyingQuery


def get_schedule(scheduler: SchedulerCore,
                 input_seq_length: int) -> Tuple[List[int], List[int], List[int], List[PipelineStage]]:
    """
    Get schedule.
    Note: this will also register the request usage in kv expectation if succeeded

    :param scheduler: scheduler
    :param input_seq_length: input sequence length
    :return: compute_node_ids (translated), start_layers (inclusive), end_layers (exclusive)
    """
    # schedule
    dummy_request = InferenceRequest(base_query_uid=None, request_uid=None, phase=RequestPhase.Initialization,
                                     token_seq_length=input_seq_length, prev_num_tokens=0, token_size=None,
                                     activation_size=None, request_creation_time=None, kv_tracker_ref=None)
    succeeded = scheduler.schedule(request=dummy_request)

    # return
    if not succeeded:
        return [], [], [], []
    else:
        compute_node_uids = []
        start_layers = []
        end_layers = []
        for stage in dummy_request.mini_pipeline[:-1]:
            compute_node_uids.append(stage.node_uid - SIMULATOR_NODE_OFFSET)
            start_layers.append(min(stage.layers_to_infer))
            end_layers.append(max(stage.layers_to_infer) + 1)
        compute_node_uids.append(0)
        start_layers.append(-1)
        end_layers.append(-1)
        return compute_node_uids, start_layers, end_layers, dummy_request.mini_pipeline


def update_scheduler(scheduler: SchedulerCore, pipeline: List[PipelineStage]) -> None:
    """
    Get schedule.
    Note: this will also register the request usage in kv expectation if succeeded

    :param scheduler: scheduler
    :param pipeline: a list of pipeline stages
    :return: compute_node_ids (translated), start_layers (inclusive), end_layers (exclusive)
    """
    dummy_request = InferenceRequest(base_query_uid=None, request_uid=None, phase=RequestPhase.Increment,
                                     token_seq_length=1, prev_num_tokens=None, token_size=None,
                                     activation_size=None, request_creation_time=None, kv_tracker_ref=None)
    dummy_request.pipeline_set = True
    dummy_request.mini_pipeline = pipeline
    scheduler.schedule(request=dummy_request)


def release_kv_expectation(scheduler: SchedulerCore, input_len: int, pipeline: List[PipelineStage]):
    route, start_layers, end_layers = [], [], []
    for pipeline_stage in pipeline[:-1]:
        route.append(pipeline_stage.node_uid)
        start_layers.append(min(pipeline_stage.layers_to_infer))
        end_layers.append(max(pipeline_stage.layers_to_infer) + 1)
    scheduler.remove_from_kv_expectation(input_seq_length=input_len,
                                         route=route,
                                         start_idx_list=start_layers,
                                         end_idx_list=end_layers)


def run_maxflow_host_online(
        # model and machine
        machine_num_dict: Dict[str, int],
        model_name: ModelName,
        # cluster
        complete_cluster_file_name: str,
        machine_profile_name: str,
        solution_file_name: str,
        simulator_cluster_file_name: str,
        real_sys_config_file_name: str,
        # throughput
        avg_throughput: float,
        duration: int,
        # result
        result_logging_dir: str,
) -> None:
    """
    Run host with !!![MaxFlow + Online mode]!!!.
    machine_num_dict: e.g.: {"A100": 4, "V100": 0, "L4": 6, "L4x2": 0, "T4": 6, "T4x2": 6, "T4x4": 2}
    """
    print("Initializing host with MaxFlow scheduling!")

    # ----------------------------------- Init Scheduler ----------------------------------- #
    # load the layout
    layout_synthesizer = LayoutSynthesizer(complete_cluster_file_name=complete_cluster_file_name,
                                           machine_profile_name=machine_profile_name,
                                           model_name=model_name,
                                           workspace_path=result_logging_dir,
                                           layout_method=LayoutMethod.LoadExisting,
                                           machine_num_dict=machine_num_dict)
    layout_args = {
        "solution_file_name": solution_file_name,
        "simulator_cluster_file_name": simulator_cluster_file_name
    }
    cluster_file_path = layout_synthesizer.synthesize(args=layout_args)

    # load the simulator, here we only use it to initialize the maxflow scheduler
    simulator = ClusterSimulator(model_name=model_name, machine_num_dict=machine_num_dict)
    simulator.from_ini_file(config_file_name=cluster_file_path)
    scheduler_args = {
        "kv_param": KVParameters(expected_kv_hwm=0.9, expected_output_length_ratio=0.6),
        "scheduling_mode": SchedulingMode.Online,
    }
    simulator.init_scheduler(scheduling_method=SchedulingMethod.MaxFlow, args=scheduler_args)
    simulator.init_query_manager()
    simulator.mark_as_ready()
    layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()

    # extract the scheduler
    maxflow_scheduler: SchedulerCore = simulator.scheduler.core
    # -------------------------------------------------------------------------------------- #

    # ------------------------------------- Online Generator ------------------------------------ #
    trace_generator = TraceGenerator(arrival_rate_source=ArrivalRateSource.AzureConv,
                                     length_dataset=Dataset.AzureConversation,
                                     cluster_token_throughput=avg_throughput, seed=0)
    trace = trace_generator.generate_trace(start_time=0, duration=duration)
    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------- Init System ------------------------------------ #
    host_ip: str = get_local_ip()
    assert host_ip.startswith("10"), "Local IP must be of form 10.xxx.xxx.xxx"
    llm_host.start_network_threads(CONFIG_BROADCAST_ADDR, host_ip, real_sys_config_file_name, "maxflow")
    time.sleep(20)
    print("[Python] Cluster initialization finished!")
    # -------------------------------------------------------------------------------------- #
    ground_zero = time.time()
    next_query_id = 0
    flying_queries_dict = {}
    # -----  log items ----- #
    # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
    query_routes = []
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    # ---------------------- #
    while True:
        # get time
        now = time.time() - ground_zero
        if now > duration + 30:
            break

        # send new requests into cluster if needed
        while not len(trace) == 0 and trace[0][0] <= now:
            # the request has a time stamp smaller than now, should be sent
            expected_submit_time, input_length, output_length = trace.pop(0)

            # schedule the request
            compute_node_uids, start_layers, end_layers, pipeline = get_schedule(scheduler=maxflow_scheduler,
                                                                                 input_seq_length=input_length)

            # get query id
            cur_query_id = next_query_id
            next_query_id += 1

            # send it into the cluster
            llm_host.launch_request(
                "prompt",  # request_type
                cur_query_id,  # request_id
                input_length,  # num_tokens
                input_length + output_length,  # max_num_tokens
                [i for i in range(input_length)],  # token_ids
                True,  # set_routing
                compute_node_uids,  # server_ids
                start_layers,  # start_layer_ids
                end_layers,  # end_layer_ids
            )

            # put into flying queries
            flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                            input_length=input_length,
                                                            output_length=output_length,
                                                            compute_node_uids=compute_node_uids,
                                                            start_layers=start_layers,
                                                            end_layers=end_layers,
                                                            pipeline=pipeline)

            # save log
            # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
            query_routes.append((cur_query_id, input_length, output_length, compute_node_uids, start_layers,
                                 end_layers))
            # time - query id - in/out - phase - context_len - this_iter_processed
            events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
            print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                  f"max_len = {input_length + output_length}")

        # get finished requests
        now = time.time() - ground_zero
        finished_query_ids, generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        for query_uid in finished_query_ids:
            # first receive the message
            py_on_the_fly_query = flying_queries_dict[query_uid]
            if py_on_the_fly_query.processed_tokens == 0:
                # prompt phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1))
                py_on_the_fly_query.processed_tokens += py_on_the_fly_query.input_length + 1
            else:
                # decode phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1))
                py_on_the_fly_query.processed_tokens += 1

            # then we decide whether to send out new messages (decodes)
            max_size = py_on_the_fly_query.input_length + py_on_the_fly_query.output_length
            assert py_on_the_fly_query.processed_tokens <= max_size, "Found request that did not end!"
            if py_on_the_fly_query.processed_tokens == max_size:
                # not send: finished, remove from expectations
                del flying_queries_dict[query_uid]
                release_kv_expectation(scheduler=maxflow_scheduler, input_len=py_on_the_fly_query.input_length,
                                       pipeline=py_on_the_fly_query.pipeline)
                print(f"Query {query_uid}, finished (total_len={py_on_the_fly_query.processed_tokens})")

            else:
                # first we update the scheduler
                update_scheduler(scheduler=maxflow_scheduler,
                                 pipeline=py_on_the_fly_query.pipeline)

                # then we send the query back into the cluster
                llm_host.launch_request(
                    "decode",  # request_type
                    query_uid,  # request_id
                    py_on_the_fly_query.processed_tokens,  # num_tokens (context size)
                    max_size,  # max_num_tokens
                    [-1],  # token_ids
                    True,  # set_routing
                    py_on_the_fly_query.compute_node_uids,  # server_ids
                    py_on_the_fly_query.start_layers,  # start_layer_ids
                    py_on_the_fly_query.end_layers,  # end_layer_ids
                )

                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1))

    # save logging files
    print(f"Queries still flying: {flying_queries_dict.keys()}.")
    query_routes_file_name = os.path.join(result_logging_dir, "query_route.txt")
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(query_routes_file_name, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    with open(events_file_name, "w") as f:
        for item in events:
            f.write(f"{item}\n")


def run_maxflow_host_offline(
        # model and machine
        machine_num_dict: Dict[str, int],
        model_name: ModelName,
        # cluster
        complete_cluster_file_name: str,
        machine_profile_name: str,
        solution_file_name: str,
        simulator_cluster_file_name: str,
        real_sys_config_file_name: str,
        # throughput
        initial_launch_num: int,
        feeding_hwm: float,
        duration: int,
        # result
        result_logging_dir: str,
) -> None:
    """
    Run host with !!![MaxFlow + Offline mode]!!!.
    machine_num_dict: e.g.: {"A100": 4, "V100": 0, "L4": 6, "L4x2": 0, "T4": 6, "T4x2": 6, "T4x4": 2}
    """
    print("Initializing host with MaxFlow scheduling!")

    # ----------------------------------- Init Scheduler ----------------------------------- #
    # load the layout
    layout_synthesizer = LayoutSynthesizer(complete_cluster_file_name=complete_cluster_file_name,
                                           machine_profile_name=machine_profile_name,
                                           model_name=model_name,
                                           workspace_path=result_logging_dir,
                                           layout_method=LayoutMethod.LoadExisting,
                                           machine_num_dict=machine_num_dict)
    layout_args = {
        "solution_file_name": solution_file_name,
        "simulator_cluster_file_name": simulator_cluster_file_name
    }
    cluster_file_path = layout_synthesizer.synthesize(args=layout_args)

    # load the simulator, here we only use it to initialize the maxflow scheduler
    simulator = ClusterSimulator(model_name=model_name, machine_num_dict=machine_num_dict)
    simulator.from_ini_file(config_file_name=cluster_file_path)
    scheduler_args = {
        "kv_param": KVParameters(expected_kv_hwm=0.85, expected_output_length_ratio=1),
        "scheduling_mode": SchedulingMode.Offline,
    }
    assert feeding_hwm <= 0.8, "Feed high water mark should be smaller than KV expected high water mark!"
    simulator.init_scheduler(scheduling_method=SchedulingMethod.MaxFlow, args=scheduler_args)
    simulator.init_query_manager()
    simulator.mark_as_ready()
    layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()

    # extract the scheduler
    maxflow_scheduler: SchedulerCore = simulator.scheduler.core
    # -------------------------------------------------------------------------------------- #

    # ------------------------------------- Offline Initial ------------------------------------- #
    length_sampler = LengthSampler(dataset=Dataset.AzureConversation, seed=0)
    initial_requests = []
    for i in range(initial_launch_num):
        request_time = 0.1 + i * 0.1
        input_length, output_length = length_sampler.sample_length()
        initial_requests.append((request_time, input_length, output_length))
    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------- Init System ------------------------------------ #
    host_ip: str = get_local_ip()
    assert host_ip.startswith("10"), "Local IP must be of form 10.xxx.xxx.xxx"
    llm_host.start_network_threads(CONFIG_BROADCAST_ADDR, host_ip, real_sys_config_file_name, "maxflow")
    time.sleep(20)
    print("[Python] Cluster initialization finished!")
    # -------------------------------------------------------------------------------------- #
    ground_zero = time.time()
    next_query_id = 0
    flying_queries_dict = {}
    last_log_time = time.time() - ground_zero
    # -----  log items ----- #
    # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
    query_routes = []
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    # ---------------------- #
    while True:
        # get time
        now = time.time() - ground_zero
        if now > duration + 1:
            break

        # log bottleneck
        if now - last_log_time >= 1:
            last_log_time = now
            print(f"(t={now}) Bottleneck kv usage: {maxflow_scheduler.kv_expectation.bottleneck_usage()}.")

        # send out initial requests into the cluster
        while not len(initial_requests) == 0 and initial_requests[0][0] <= now:
            # the request has a time stamp smaller than now, should be sent
            expected_submit_time, input_length, output_length = initial_requests[0]

            # schedule the request
            compute_node_uids, start_layers, end_layers, pipeline = get_schedule(scheduler=maxflow_scheduler,
                                                                                 input_seq_length=input_length)

            # = 0 means can not schedule due to kv cache on some path is full
            if not len(compute_node_uids) == 0:
                # pop
                initial_requests.pop(0)

                # get query id
                cur_query_id = next_query_id
                next_query_id += 1

                # send it into the cluster
                llm_host.launch_request(
                    "prompt",  # request_type
                    cur_query_id,  # request_id
                    input_length,  # num_tokens
                    input_length + output_length,  # max_num_tokens
                    [i for i in range(input_length)],  # token_ids
                    True,  # set_routing
                    compute_node_uids,  # server_ids
                    start_layers,  # start_layer_ids
                    end_layers,  # end_layer_ids
                )

                # put into flying queries
                flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                                input_length=input_length,
                                                                output_length=output_length,
                                                                compute_node_uids=compute_node_uids,
                                                                start_layers=start_layers,
                                                                end_layers=end_layers,
                                                                pipeline=pipeline)

                # save log
                # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                query_routes.append((cur_query_id, input_length, output_length, compute_node_uids, start_layers,
                                     end_layers))
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
                print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                      f"max_len = {input_length + output_length}")

        # get finished requests
        now = time.time() - ground_zero
        finished_query_ids, generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        for query_uid in finished_query_ids:
            # first receive the message
            py_on_the_fly_query = flying_queries_dict[query_uid]
            if py_on_the_fly_query.processed_tokens == 0:
                # prompt phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1))
                py_on_the_fly_query.processed_tokens += py_on_the_fly_query.input_length + 1

                # at the end of prompt phase, we have a choice to add more requests if kv cache is ok
                current_bottleneck = maxflow_scheduler.kv_expectation.bottleneck_usage()
                if current_bottleneck <= feeding_hwm:
                    # launch a new request
                    # the request has a time stamp smaller than now, should be sent
                    input_length, output_length = length_sampler.sample_length()

                    # schedule the request
                    compute_node_uids, start_layers, end_layers, pipeline = get_schedule(scheduler=maxflow_scheduler,
                                                                                         input_seq_length=input_length)

                    # = 0 means can not schedule due to kv cache on some path is full
                    if not len(compute_node_uids) == 0:
                        # get query id
                        cur_query_id = next_query_id
                        next_query_id += 1

                        # send it into the cluster
                        llm_host.launch_request(
                            "prompt",  # request_type
                            cur_query_id,  # request_id
                            input_length,  # num_tokens
                            input_length + output_length,  # max_num_tokens
                            [i for i in range(input_length)],  # token_ids
                            True,  # set_routing
                            compute_node_uids,  # server_ids
                            start_layers,  # start_layer_ids
                            end_layers,  # end_layer_ids
                        )

                        # put into flying queries
                        flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                                        input_length=input_length,
                                                                        output_length=output_length,
                                                                        compute_node_uids=compute_node_uids,
                                                                        start_layers=start_layers,
                                                                        end_layers=end_layers,
                                                                        pipeline=pipeline)

                        # save log
                        # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                        query_routes.append((cur_query_id, input_length, output_length, compute_node_uids, start_layers,
                                             end_layers))
                        # time - query id - in/out - phase - context_len - this_iter_processed
                        events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
                        print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                              f"max_len = {input_length + output_length} (prompt request more)")
                    else:
                        print(f"Scheduler rejected a new request at end of prompt "
                              f"(input_len={input_length}, output_len={output_length})")

            else:
                # decode phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1))
                py_on_the_fly_query.processed_tokens += 1

            # then we decide whether to send out new messages (decodes)
            max_size = py_on_the_fly_query.input_length + py_on_the_fly_query.output_length
            assert py_on_the_fly_query.processed_tokens <= max_size, "Found request that did not end!"
            if py_on_the_fly_query.processed_tokens == max_size:
                # not send: finished, remove from expectations
                del flying_queries_dict[query_uid]
                release_kv_expectation(scheduler=maxflow_scheduler, input_len=py_on_the_fly_query.input_length,
                                       pipeline=py_on_the_fly_query.pipeline)
                print(f"Query {query_uid}, finished (total_len={py_on_the_fly_query.processed_tokens})")

                # we have a chance to determine whether we need to add a new request
                current_bottleneck = maxflow_scheduler.kv_expectation.bottleneck_usage()
                if current_bottleneck <= feeding_hwm:
                    # launch a new request
                    # the request has a time stamp smaller than now, should be sent
                    input_length, output_length = length_sampler.sample_length()

                    # schedule the request
                    compute_node_uids, start_layers, end_layers, pipeline = get_schedule(scheduler=maxflow_scheduler,
                                                                                         input_seq_length=input_length)

                    # = 0 means can not schedule due to kv cache on some path is full
                    if not len(compute_node_uids) == 0:
                        # get query id
                        cur_query_id = next_query_id
                        next_query_id += 1

                        # send it into the cluster
                        llm_host.launch_request(
                            "prompt",  # request_type
                            cur_query_id,  # request_id
                            input_length,  # num_tokens
                            input_length + output_length,  # max_num_tokens
                            [i for i in range(input_length)],  # token_ids
                            True,  # set_routing
                            compute_node_uids,  # server_ids
                            start_layers,  # start_layer_ids
                            end_layers,  # end_layer_ids
                        )

                        # put into flying queries
                        flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                                        input_length=input_length,
                                                                        output_length=output_length,
                                                                        compute_node_uids=compute_node_uids,
                                                                        start_layers=start_layers,
                                                                        end_layers=end_layers,
                                                                        pipeline=pipeline)

                        # save log
                        # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                        query_routes.append((cur_query_id, input_length, output_length, compute_node_uids, start_layers,
                                             end_layers))
                        # time - query id - in/out - phase - context_len - this_iter_processed
                        events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
                        print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                              f"max_len = {input_length + output_length} (decode finish request replacement)")
                    else:
                        print(f"Scheduler rejected a new request at end of decode "
                              f"(input_len={input_length}, output_len={output_length})")

            else:
                # first we update the scheduler
                update_scheduler(scheduler=maxflow_scheduler,
                                 pipeline=py_on_the_fly_query.pipeline)

                # then we send the query back into the cluster
                llm_host.launch_request(
                    "decode",  # request_type
                    query_uid,  # request_id
                    py_on_the_fly_query.processed_tokens,  # num_tokens (context size)
                    max_size,  # max_num_tokens
                    [-1],  # token_ids
                    True,  # set_routing
                    py_on_the_fly_query.compute_node_uids,  # server_ids
                    py_on_the_fly_query.start_layers,  # start_layer_ids
                    py_on_the_fly_query.end_layers,  # end_layer_ids
                )

                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1))

    # save logging files
    print(f"Queries still flying: {flying_queries_dict.keys()}.")
    query_routes_file_name = os.path.join(result_logging_dir, "query_route.txt")
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(query_routes_file_name, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    with open(events_file_name, "w") as f:
        for item in events:
            f.write(f"{item}\n")
