# 2024.01.31 Yixuan Mei

import copy

from typing import Dict, List, Tuple, TYPE_CHECKING

from simulator.event_simulator.utils import BASE_QUERY_UID
from simulator.event_simulator.kv_cache import KVTracker
from simulator.event_simulator.request import InferenceRequest, RequestPhase, PipelineStage
from simulator.event_simulator.latency_analyzer import LatencyAnalyzer

if TYPE_CHECKING:
    from simulator.event_simulator.cluster_simulator import ClusterSimulator


class QueryInferenceHistory:
    def __init__(self, request_uid: int, request_phase: RequestPhase, start_time: float, end_time: float,
                 token_seq_length: int, prev_num_tokens: int, path: List[PipelineStage]) -> None:
        """
        QueryInferenceHistory describes how one iteration of the inference happens.

        :param request_uid: uid of the request for this iteration
        :param request_phase: phase of the request for this iteration
        :param start_time: start time of this iteration (when send to source)
        :param end_time: end time of this interation (when arrive at sink)
        :param token_seq_length: token sequence length active in this iteration
        :param prev_num_tokens: number of previous tokens in the query before this iteration
        :param path: the path used in this interation
        :return: None
        """
        self.request_uid: int = request_uid
        self.request_phase: RequestPhase = request_phase
        self.start_time: float = start_time
        self.end_time: float = end_time
        self.token_seq_length: int = token_seq_length
        self.prev_num_tokens: int = prev_num_tokens
        self.path: List[PipelineStage] = path


class Query:
    def __init__(self, query_uid: int, creation_time: float, input_seq_length: int, output_seq_length: int,
                 total_num_layers: int) -> None:
        """
        A query to the large language model.
        Note: 1. we need output_seq_length to determine how many iterations this query will be sent for inference
              2. each query will be decomposed into a list of inference requests and sent into cluster for inference

        :param query_uid: uid of this query
        :param creation_time: creation time
        :param input_seq_length: input sequence length
        :param output_seq_length: expected output sequence length
        :param total_num_layers: total number of layers that will be inferred
        :return: None
        """
        # basic information
        self.query_uid: int = query_uid
        self.creation_time: float = creation_time
        self.input_seq_length: int = input_seq_length
        self.output_seq_length: int = output_seq_length

        # inference information
        self.inference_history: List[QueryInferenceHistory] = []
        self.next_iteration_idx: int = 0
        self.inferred_token_count: int = 0

        # kv tracker
        self.kv_tracker: KVTracker = KVTracker(total_num_layers=total_num_layers)

    def get_next_iteration(self) -> Tuple[RequestPhase, int, int] or Tuple[None, None, None]:
        """
        Get the next iteration of this request. If the query has already inferred the last iteration,
        return None.

        :return: request phase, token_seq_length, inferred num tokens / None, None, None
        """
        assert self.next_iteration_idx == len(self.inference_history), "Last iteration has not finished!"
        if self.next_iteration_idx == 0:
            self.next_iteration_idx += 1
            return RequestPhase.Initialization, self.input_seq_length, self.inferred_token_count
        elif 1 <= self.next_iteration_idx <= self.output_seq_length:
            self.next_iteration_idx += 1
            return RequestPhase.Increment, 1, self.inferred_token_count
        else:
            assert self.next_iteration_idx == self.output_seq_length + 1, "Bad next iteration index found!"
            return None, None, None

    def submit_finished_request(self, request: InferenceRequest) -> None:
        """
        Submit a finished inference request into this query.

        :param request: the request to submit
        :return: None
        """
        assert request.base_query_uid == self.query_uid, "Try to submit a request that does not belong!"

        # increase inferred token count
        self.inferred_token_count += request.token_seq_length

        # build the RequestInferenceHistory entry
        request_start_time: float = request.location_history[0][1]
        request_end_time: float = request.location_history[-1][1]
        self.inference_history.append(QueryInferenceHistory(request_uid=request.request_uid,
                                                            request_phase=request.phase,
                                                            start_time=request_start_time,
                                                            end_time=request_end_time,
                                                            prev_num_tokens=request.prev_num_tokens,
                                                            token_seq_length=request.token_seq_length,
                                                            path=request.mini_pipeline))


class QueryManagerParameters:
    def __init__(self, token_size: float, token_activation_size: float, total_num_layers: int) -> None:
        """
        Parameters used in query manager.

        :param token_size: size to store a token
        :param token_activation_size: size to store activation for a token
        :param total_num_layers: total number of layers in this model
        :return: None
        """
        self.token_size: float = token_size
        self.token_activation_size: float = token_activation_size
        self.total_num_layers: int = total_num_layers


class QueryManager:
    def __init__(self, param: QueryManagerParameters, simulator: "ClusterSimulator") -> None:
        """
        A manager for queries

        :param param: parameters of the query manager
        :return: None
        """
        # parameters
        self.param: QueryManagerParameters = param
        self.simulator: "ClusterSimulator" = simulator

        # query tracking
        self.next_query_uid: int = BASE_QUERY_UID
        self.queries_on_the_fly: Dict[int, Query] = {}
        self.finished_queries: Dict[int, Tuple[float, Query]] = {}
        self.latency_analyzer: LatencyAnalyzer = LatencyAnalyzer()

    def get_next_query_uid(self) -> int:
        """
        Get next query uid.

        :return: next query uid
        """
        next_query_uid = self.next_query_uid
        self.next_query_uid += 1
        return next_query_uid

    def issue_query(self, creation_time: float, input_seq_length: int, output_seq_length: int) -> None:
        """
        Issue a new query into the cluster

        :param creation_time: time when this query is created
        :param input_seq_length: input sequence length
        :param output_seq_length: expected output sequence length
        :return: None
        """
        # construct the new query
        assert creation_time >= self.simulator.current_time, "Can not create queries for the past!"
        new_query: Query = Query(query_uid=self.get_next_query_uid(),
                                 creation_time=creation_time,
                                 input_seq_length=input_seq_length,
                                 output_seq_length=output_seq_length,
                                 total_num_layers=self.param.total_num_layers)
        self.queries_on_the_fly[new_query.query_uid] = new_query

        # issue the first iteration into the cluster
        next_phase, next_token_seq_length, inferred_token_count = new_query.get_next_iteration()
        self.simulator.issue_command_new_request(base_query_uid=new_query.query_uid,
                                                 arrive_time=creation_time,
                                                 phase=next_phase,
                                                 token_seq_length=next_token_seq_length,
                                                 prev_num_tokens=inferred_token_count,
                                                 token_size=self.param.token_size,
                                                 activation_size=self.param.token_activation_size,
                                                 pipeline=None,
                                                 kv_tracker_ref=new_query.kv_tracker)

    def reject_query(self, request: InferenceRequest):
        """
        Reject a query because of failed scheduling (due to kv cache).
        Note: 1. can only do so in offline mode
              2. can only do so with GlobalFlowScheduler
              3. can only do so for initialization phase requests

        :param request:
        :return:
        """
        from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler, SchedulingMode
        assert isinstance(self.simulator.scheduler, GlobalFlowScheduler), "Can only reject in GlobalFlowScheduler!"
        assert self.simulator.scheduler.scheduling_mode == SchedulingMode.Offline, \
            "Can only reject a query in offline mode!"
        assert request.phase == RequestPhase.Initialization, "Can only reject initialization phase requests!"
        base_query_uid = request.base_query_uid
        del self.queries_on_the_fly[base_query_uid]
        # FIXME: restore the log here
        # print(f"A query is rejected due to low kv-cache in specific routes!")

    def collect_finished_request(self, current_time: float, request: InferenceRequest) -> bool:
        """
        Collect a finished inference request (i.e. an iteration of some query) and update the corresponding query.
        Note: this function is called by the SinkNode (i.e. coordinator of the simulator)

        :param current_time: time when this request finishes
        :param request: the request to collect
        :return: whether the query has finished or not
        """
        # collect the request and update the query
        assert current_time == request.location_history[-1][1], "Time mismatch!"
        target_query: Query = self.queries_on_the_fly[request.base_query_uid]
        target_query.submit_finished_request(request=request)

        # log current request's latency
        self.latency_analyzer.add_request(request=request)

        # issue new iteration of that query (if last iteration has finished, move to finished)
        next_phase, next_token_seq_length, inferred_token_count = target_query.get_next_iteration()
        if next_phase is not None:
            # issue next iteration
            self.simulator.issue_command_new_request(base_query_uid=target_query.query_uid,
                                                     arrive_time=current_time,
                                                     phase=next_phase,
                                                     token_seq_length=next_token_seq_length,
                                                     prev_num_tokens=inferred_token_count,
                                                     token_size=self.param.token_size,
                                                     activation_size=self.param.token_activation_size,
                                                     pipeline=copy.deepcopy(request.mini_pipeline),
                                                     kv_tracker_ref=target_query.kv_tracker)
            return False
        else:
            # move query to finished
            assert (target_query.inferred_token_count == target_query.input_seq_length +
                    target_query.output_seq_length), "Found unfinished query!"
            self.finished_queries[target_query.query_uid] = (current_time, target_query)
            self.latency_analyzer.add_query(query=target_query)
            del self.queries_on_the_fly[target_query.query_uid]

            # remove query from scheduler's kv expectation if we are using MaxFlow Scheduling
            from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler
            if isinstance(self.simulator.scheduler, GlobalFlowScheduler):
                nodes_used_in_pipeline: List[int] = []
                start_layer_idx_list: List[int] = []
                end_layer_idx_list: List[int] = []
                for pipeline_stage in target_query.inference_history[0].path[:-1]:
                    nodes_used_in_pipeline.append(pipeline_stage.node_uid)
                    start_layer_idx_list.append(pipeline_stage.layers_to_infer[0])
                    end_layer_idx_list.append(pipeline_stage.layers_to_infer[-1] + 1)
                self.simulator.scheduler.core.remove_from_kv_expectation(
                    input_seq_length=target_query.input_seq_length,
                    route=nodes_used_in_pipeline,
                    start_idx_list=start_layer_idx_list,
                    end_idx_list=end_layer_idx_list
                )

            # return
            return True
