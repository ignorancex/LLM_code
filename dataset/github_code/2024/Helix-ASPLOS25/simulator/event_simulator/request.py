# 2023.12.11 Yixuan Mei

from enum import Enum
from typing import List, Tuple, TYPE_CHECKING

from simulator.event_simulator.base_node import NodeType
from simulator.event_simulator.kv_cache import KVTracker

if TYPE_CHECKING:
    from simulator.event_simulator.network_link import NetworkLink


class RoutingHistoryEntry:
    def __init__(self, link_uid: int, in_node_uid: int, out_node_uid: int, bandwidth_usage: float) -> None:
        """
        Routing history entry

        :param link_uid: the link used at this hop
        :param in_node_uid: in-node uid
        :param out_node_uid: out-node uid
        :param bandwidth_usage: bandwidth usage on this link
        :return: None
        """
        self.link_uid: int = link_uid
        self.in_node_uid: int = in_node_uid
        self.out_node_uid: int = out_node_uid
        self.bandwidth_usage: float = bandwidth_usage


class InferenceHistoryEntry:
    def __init__(self, layer_id: int, node_uid: int) -> None:
        """
        Inference history entry.

        :param layer_id: which layer of the model gets inferred
        :param node_uid: node uid
        :return: None
        """
        self.layer_id: int = layer_id
        self.node_uid: int = node_uid


class RequestPhase(Enum):
    """ initialization phase / increment phase """
    Initialization = "InferenceRequestPhase.Initialization"
    Increment = "InferenceRequestPhase.Increment"


class RequestLocation(Enum):
    """ location of a request """
    SourceNode = "RequestLocation.SourceNode"
    SinkNode = "RequestLocation.SinkNode"
    ComputeNode = "RequestLocation.ComputeNode"
    Link = "RequestLocation.Link"


class PipelineStage:
    def __init__(self, link_uid: int, bandwidth_usage: float, node_uid: int, layers_to_infer: List[int]) -> None:
        """
        A pipeline stage in scheduling.

        :param link_uid: uid of the link to use
        :param bandwidth_usage: bandwidth usage on this link
        :param node_uid: uid of the node to use
        :param layers_to_infer: layers to infer on this node
        """
        self.link_uid: int = link_uid
        self.bandwidth_usage: float = bandwidth_usage
        self.node_uid: int = node_uid
        self.layers_to_infer: List[int] = layers_to_infer


class InferenceRequest:
    def __init__(self, base_query_uid: int, request_uid: int, phase: RequestPhase, token_seq_length: int,
                 prev_num_tokens: int, token_size: float, activation_size: float, request_creation_time: float,
                 kv_tracker_ref: KVTracker) -> None:
        """
        An inference request (batch_size = 1).
        Our performance modeling is based on: https://arxiv.org/pdf/2305.02440.pdf
        Token sequence length must be 1 for increment phase. token_size is usually # tokens * 4 Byte.
        activation_size is usually # tokens * 8k * 2 Byte.

        :param request_uid: unique identifier of this request
        :param phase: which phase is the current request in (initialization / increment)
        :param token_seq_length: how many tokens this request contains (increment must be 1)
        :param prev_num_tokens: number of previous tokens in the query before this iteration
        :param token_size: how much space token transmission needs
        :param activation_size: how much space activation transmission needs
        :param request_creation_time: time of creation for current request
        :param kv_tracker_ref: a reference to the kv tracker in base query
        :return: None
        """
        # basic properties
        self.base_query_uid: int = base_query_uid
        self.request_uid: int = request_uid
        self.phase: RequestPhase = phase
        self.token_seq_length: int = token_seq_length
        self.prev_num_tokens: int = prev_num_tokens
        self.token_size: float = token_size
        self.activation_size: float = activation_size
        assert phase == RequestPhase.Initialization or token_seq_length == 1, "Increment phase must have seq length 1!"
        assert phase == RequestPhase.Increment or prev_num_tokens == 0, "Initialization phase should have 0 prev tokens"

        # routing and inference history
        self.routing_history: List[RoutingHistoryEntry] = []
        self.inference_history: List[InferenceHistoryEntry] = []

        # location of a request
        self.current_location: RequestLocation = RequestLocation.SourceNode
        self.current_location_uid: int = 0
        self.location_history: List[Tuple[str, float]] = [(f"{self.current_location}-{self.current_location_uid}",
                                                           request_creation_time)]

        # global routing (Used by Global MaxFlow Scheduler)
        self.pipeline_set: bool = False
        self.mini_pipeline: List[PipelineStage] = []
        self.current_pipeline_stage_idx: int = -1

        # kv cache tracking (a reference to the tracker in base query)
        self.kv_tracker_ref: KVTracker = kv_tracker_ref

    def get_description(self) -> str:
        """
        Get description of current request.

        :return: description string
        """
        attributes = {"request_uid": self.request_uid,
                      "phase": self.phase,
                      "token_seq_length": self.token_seq_length,
                      "token_size": self.token_size,
                      "activation_size": self.activation_size,
                      "current_location": f"{self.current_location}",
                      "current_location_uid": f"{self.current_location_uid}"}
        return f"{attributes}"

    def add_routing_history(self, link: "NetworkLink", bandwidth_usage: float) -> None:
        """
        Add routing history for current request.

        :param link: the link that this request just passed
        :param bandwidth_usage: how much bandwidth is used when traversing this link
        :return: None
        """
        # check that the routing is consistent
        if len(self.routing_history) == 0:
            assert link.node_in_type == NodeType.Source
        else:
            assert self.routing_history[-1].out_node_uid == link.node_in.node_uid

        # add routing history
        self.routing_history.append(RoutingHistoryEntry(link_uid=link.link_uid,
                                                        in_node_uid=link.node_in.node_uid,
                                                        out_node_uid=link.node_out.node_uid,
                                                        bandwidth_usage=bandwidth_usage))

    def add_inference_history(self, layers: List[int], node_uid: int) -> None:
        """
        Add given layers into inference history

        :param layers: layers that are inferred
        :param node_uid: on which node are these layers inferred
        :return: None
        """
        # we check consistency while appending to history
        _last_layer_inferred = -1 if len(self.inference_history) == 0 else self.inference_history[-1].layer_id
        for layer_id in layers:
            assert layer_id == _last_layer_inferred + 1, "Inconsistent layer id found!"
            self.inference_history.append(InferenceHistoryEntry(layer_id=layer_id, node_uid=node_uid))
            _last_layer_inferred += 1

    def update_location(self, new_location: RequestLocation, new_location_uid: int, arrive_time: float) -> None:
        """
        Update location of a request.

        :param new_location: which entity is the request on currently
        :param new_location_uid: unique identifier of that entity
        :param arrive_time: when the request arrives at the new location
        :return: None
        """
        self.current_location = new_location
        self.current_location_uid = new_location_uid
        self.location_history.append((f"{self.current_location}-{self.current_location_uid}", arrive_time))

    def get_last_node_and_link_uid(self) -> (int, int):
        """
        Get uid of the last node and link this request passes. Must be called when this request is on a node.

        :return: last node uid, last link uid
        """
        # check current request location (this ensures that there are at least two entries in location history)
        assert self.current_location == RequestLocation.ComputeNode, "Can not get last node and link uid here!"

        # extract uids from location history
        last_node_name: str = self.location_history[-3][0]
        last_link_name: str = self.location_history[-2][0]
        last_node_uid: int = int(str.split(last_node_name, "-")[1])
        last_link_uid: int = int(str.split(last_link_name, "-")[1])
        return last_node_uid, last_link_uid

    def get_num_layers_on_node(self, node_uid: int) -> int:
        """
        Get number of layers inferred on a given node.

        :param node_uid: uid of the node
        :return: number of layers inferred on this node
        """
        num_layers: int = 0
        for inference_history_entry in self.inference_history:
            if inference_history_entry.node_uid == node_uid:
                num_layers += 1
        return num_layers

    def set_pipeline(self, pipeline: List[PipelineStage]) -> None:
        """
        Set mini pipeline for this request.

        :param pipeline: the pipeline to go
        :return: None
        """
        assert not self.pipeline_set, "Pipeline already set!"
        self.mini_pipeline = pipeline
        self.pipeline_set = True

    def add_pipeline_stage(self, pipeline_stage: PipelineStage) -> None:
        """
        Add a new stage to the mini pipeline for this request.

        :param pipeline_stage: a new stage in the pipeline
        :return: None
        """
        self.mini_pipeline.append(pipeline_stage)

    def mark_pipeline_set(self) -> None:
        """
        Mark pipeline as set

        :return: None
        """
        self.pipeline_set = True

    def get_current_pipeline_stage(self) -> PipelineStage:
        """
        Get current pipeline stage.

        :return: current pipeline stage
        """
        assert self.pipeline_set, "No pipeline found on request!"
        assert not self.current_pipeline_stage_idx == -1, "At source node, before first stage!"
        return self.mini_pipeline[self.current_pipeline_stage_idx]

    def get_next_pipeline_stage(self) -> PipelineStage:
        """
        Get the next pipeline stage.

        :return: the next pipeline stage
        """
        assert self.pipeline_set, "No pipeline found on request!"
        assert self.current_pipeline_stage_idx + 1 < len(self.mini_pipeline), "Already in the last stage!"
        return self.mini_pipeline[self.current_pipeline_stage_idx + 1]

    def march_pipeline_stage(self) -> None:
        """
        March pipeline stage forward by one stage.

        :return: None
        """
        assert self.pipeline_set, "No pipeline found on request!"
        assert self.current_pipeline_stage_idx + 1 < len(self.mini_pipeline), "Already in the last stage!"
        self.current_pipeline_stage_idx += 1
