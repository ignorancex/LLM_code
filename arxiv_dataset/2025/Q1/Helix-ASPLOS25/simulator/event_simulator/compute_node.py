# 2023.12.12 Yixuan Mei

from typing import Dict, List, Set, Any, Tuple

from simulator.event_simulator.base_node import BaseNode, NodeType
from simulator.event_simulator.model import ModelLayer, ModelStatus
from simulator.event_simulator.kv_cache import KVCache, ActivationBackupCache
from simulator.event_simulator.network_link import NetworkLink, TransmissionObject
from simulator.event_simulator.request import InferenceRequest, RequestPhase
from simulator.event_simulator.utils import gbps


class InferenceBatch:
    def __init__(self, requests: List[InferenceRequest], duration: float, vram_usage: float) -> None:
        """
        Represent a batch of requests being inferred.

        :param requests: the list of requests being inferred
        :param duration: how long current inference takes
        :param vram_usage: how much vram current inference uses
        :return: None
        """
        self.requests: List[InferenceRequest] = requests
        self.duration: float = duration
        self.vram_usage: float = vram_usage

    def get_handle(self) -> int:
        """
        Get a handle of this InferenceBatch. We use uid of the first request as the handle. (This guarantees
        that there won't be duplicate handles within any node)

        :return: handle of this InferenceBatch
        """
        assert not len(self.requests) == 0, "Can not return handle for empty InferenceBatch!"
        return self.requests[0].request_uid


class InferenceSettings:
    def __init__(self, prompt_max_requests: int, prompt_max_tokens: int,
                 prompt_typical_requests: float, prompt_typical_tokens: int,
                 decode_max_context: int, decode_max_tokens: int,
                 decode_typical_tokens: int) -> None:
        """
        Inference settings for a node. The values are dependent on:
            1. GPU on node
            2. LLM model
            3. number of layers on node
        prompt_max_requests, prompt_max_tokens:
            1. used in execution_policy for prompt phase scheduling
        prompt_typical_request, prompt_typical_tokens:
            1. used for performance estimation (get_typical_throughput of ComputeNode and ModelManager)
            2. notice that prompt_typical_request might be smaller than 1
        decode_max_context, decode_max_tokens:
            1. used in execution_policy for decode phase scheduling
        decode_typical_tokens:
            1. used for performance estimation (get_typical_throughput of ComputeNode and ModelManager)

        :param prompt_max_requests: max number of requests in each batch (prompt phase)
        :param prompt_max_tokens: max number of tokens in each batch (prompt phase)
        :param prompt_typical_requests: typical number of requests in each batch (prompt phase, can be < 1)
        :param prompt_typical_tokens: typical number of tokens in each batch (prompt phase)
        :param decode_max_context: max number of tokens in decode phase's context
        :param decode_max_tokens: max number of tokens in each batch (decode phase), this is equal to the max
            number requests in the batch
        :param decode_typical_tokens: typical number of tokens in each batch (decode phase)
        :return: None
        """
        # some sanity checks
        assert decode_typical_tokens <= decode_max_tokens, "Typical tokens in decode phase should be less than max!"

        # prompt
        self.prompt_max_requests: int = prompt_max_requests
        self.prompt_max_tokens: int = prompt_max_tokens
        self.prompt_typical_requests: float = prompt_typical_requests
        self.prompt_typical_tokens: int = prompt_typical_tokens

        # decode
        self.decode_max_context: int = decode_max_context
        self.decode_max_tokens: int = decode_max_tokens
        self.decode_typical_tokens: int = decode_typical_tokens

    def get_description(self) -> str:
        """
        Return a description of current settings.

        :return: description str
        """
        attributes: Dict[str, Any] = {
            # prompt
            "prompt_max_requests": self.prompt_max_requests,
            "prompt_max_tokens": self.prompt_max_tokens,
            "prompt_typical_requests": self.prompt_typical_requests,
            "prompt_typical_tokens": self.prompt_typical_tokens,
            # decode
            "decode_max_context": self.decode_max_context,
            "decode_max_tokens": self.decode_max_tokens,
            "decode_typical_tokens": self.decode_typical_tokens
        }
        return f"{attributes}"


class ComputeNode(BaseNode):
    def __init__(self, node_uid: int, vram_size: float, inbound_nic_speed: float, outbound_nic_speed: float,
                 disk_speed: float, machine_type: str, kv_cache_capacity: int, activation_backup_capacity: int) -> None:
        """
        Abstraction of a GPU compute node in the cluster

        :param node_uid: unique identifier of this node
        :param vram_size: total memory size of GPUs on this node
        :param inbound_nic_speed: network speed of receiving packages
        :param outbound_nic_speed: network speed of sending packages
        :param disk_speed: disk speed for loading models
        :param machine_type: type of this machine (e.g. A100, T4, etc.)
        :param kv_cache_capacity: how many tokens can be stored in the kv cache on this node
        :param activation_backup_capacity: how many tokens can be stored in the activation backup cache on this node
        :returns: None
        """
        # basic info
        super().__init__(node_uid=node_uid, node_type=NodeType.Compute)
        self.vram_size: float = vram_size
        self.inbound_nic_speed: float = inbound_nic_speed
        self.outbound_nic_speed: float = outbound_nic_speed
        self.disk_speed: float = disk_speed
        self.machine_type: str = machine_type

        # vram and model related
        self.available_vram: float = vram_size
        self.model_status: ModelStatus = ModelStatus.NoModel
        self.in_vram_model_layers: Dict[int, ModelLayer] = {}
        self.new_model_layers: Dict[int, ModelLayer] or None = None
        self.inference_settings: InferenceSettings or None = None
        self.new_inference_settings: InferenceSettings or None = None
        self.request_uids_to_wait: Set[int] or None = None

        # network management
        # nic and connections
        self.inbound_links: Dict[int, NetworkLink] = {}
        self.inbound_available_bandwidth: float = inbound_nic_speed
        self.inbound_requests_on_the_fly: Dict[str, TransmissionObject] = {}
        self.outbound_links: Dict[int, NetworkLink] = {}
        self.outbound_available_bandwidth: float = outbound_nic_speed
        self.outbound_requests_on_the_fly: Dict[str, TransmissionObject] = {}
        # local inbound and outbound request queue
        self.inbound_request_queue: List[InferenceRequest] = []
        self.outbound_request_dict: Dict[int, InferenceRequest] = {}

        # model inference
        self.current_inference_batch: InferenceBatch or None = None
        self.current_layer_id: int = -1
        self.between_layer_queues: Dict[Tuple[int, int], List[InferenceRequest]] = {}

        # kv cache and activation backup cache
        self.kv_cache_capacity: int = kv_cache_capacity
        self.kv_cache: KVCache or None = None
        self.activation_backup_cache_capacity: int = activation_backup_capacity
        self.activation_backup_cache: ActivationBackupCache or None = None

        # overhead modeling
        self.cpu_buffer: Dict[int, InferenceRequest] = {}

        # logging
        self.entity_name: str = f"Compute-{self.node_uid}"
        self.inferred_request_uids: Set[int] = set()
        self.link_request_counters: Dict[int, int] = {}
        self.link_token_counters: Dict[int, int] = {}
        self.link_backup_request_counters: Dict[int, int] = {}
        self.link_backup_token_counters: Dict[int, int] = {}

    def add_inbound_link(self, inbound_link: NetworkLink) -> None:
        """
        Add an inbound link to this node.

        :param inbound_link: the link to be added
        :return: None
        """
        # a few topology checks
        assert inbound_link.link_uid not in self.inbound_links
        assert inbound_link.link_uid not in self.outbound_links
        assert inbound_link.node_out_type == NodeType.Compute
        assert inbound_link.node_out.node_uid == self.node_uid
        assert not inbound_link.node_in.node_uid == self.node_uid

        # add the link
        self.inbound_links[inbound_link.link_uid] = inbound_link

    def add_outbound_link(self, outbound_link: NetworkLink) -> None:
        """
        Add an outbound link to this node.

        :param outbound_link: the link to be added
        :return: None
        """
        # a few topology checks
        assert outbound_link.link_uid not in self.outbound_links
        assert outbound_link.link_uid not in self.inbound_links
        assert outbound_link.node_in_type == NodeType.Compute
        assert outbound_link.node_in.node_uid == self.node_uid
        assert not outbound_link.node_out.node_uid == self.node_uid

        # add the link
        self.outbound_links[outbound_link.link_uid] = outbound_link
        self.link_request_counters[outbound_link.link_uid] = 0
        self.link_token_counters[outbound_link.link_uid] = 0
        self.link_backup_request_counters[outbound_link.link_uid] = 0
        self.link_backup_token_counters[outbound_link.link_uid] = 0

    def prepare_loading_model(self, new_model_layers: Dict[int, ModelLayer], request_uids_to_wait: List[int],
                              new_inference_settings: InferenceSettings) -> bool:
        """
        Load layers into this node. The model status will change to Flushing after calling this function.
        This function returns whether we can start loading right away.

        :param new_model_layers: a dict of layer id -> ModelLayer, representing the new model
        :param request_uids_to_wait: a list of requests to wait before actually load the model
        :param new_inference_settings: new inference settings
        :return: whether all dependent requests are cleared
        """
        # set the model status to flushing
        assert self.model_status == ModelStatus.NoModel or self.model_status == ModelStatus.Ready
        self.model_status = ModelStatus.Flushing

        # store the new layers (this might be more than what we need to load if there is overlap)
        self.new_model_layers = new_model_layers
        self.new_inference_settings = new_inference_settings

        # determine which request to wait on (we might have already inferred some of the requests)
        self.request_uids_to_wait = set()
        for request_uid in request_uids_to_wait:
            if request_uid not in self.inferred_request_uids:
                self.request_uids_to_wait.add(request_uid)

        # a sanity check: all requests already issued to this node must be waited on
        for request in self.inbound_request_queue:
            assert request.request_uid in self.request_uids_to_wait, "Found request not waited-on but should be!"

        # return whether we can start loading right away
        return len(self.request_uids_to_wait) == 0

    def ready_to_load_model(self) -> bool:
        """
        Whether the node is ready to load model.

        :return: Whether the node is ready to load model.
        """
        return self.model_status == ModelStatus.Flushing and len(self.request_uids_to_wait) == 0

    def start_loading_model(self) -> float:
        """
        Start loading model.

        :return: time to finish
        """
        # check model status
        assert self.ready_to_load_model(), "Model must be ready to load!"
        assert self.current_inference_batch is None, "Node must be idle to load model!"
        assert self.new_model_layers is not None, "New model does not exist!"

        # calculate new model size and loading time
        total_vram_size: float = 0
        loading_size: float = 0
        for layer_id in self.new_model_layers:
            total_vram_size += self.new_model_layers[layer_id].vram_usage
            if layer_id not in self.in_vram_model_layers:
                loading_size += self.new_model_layers[layer_id].vram_usage
        assert total_vram_size <= self.vram_size, "Try to load a model that does not fit into vram!"
        loading_time: float = loading_size / self.disk_speed

        # set model status to loading
        self.model_status = ModelStatus.Loading

        # return duration
        return loading_time

    def finish_loading_model(self) -> None:
        """
        Finish loading model.

        :return: None
        """
        # check model status
        assert self.model_status == ModelStatus.Loading, "Bad model status!"

        # set model and model status
        self.model_status = ModelStatus.Ready
        self.in_vram_model_layers = self.new_model_layers
        self.inference_settings = self.new_inference_settings
        self.new_model_layers = None
        self.new_inference_settings = None
        self.request_uids_to_wait = None

        # initialize inference-related data structures
        self.current_layer_id = min(self.in_vram_model_layers.keys())
        all_layer_ids = sorted(list(self.in_vram_model_layers.keys()))
        between_layer_queues: Dict[Tuple[int, int], List[InferenceRequest]] = {}
        for idx in range(len(all_layer_ids) - 1):
            between_layer_queues[(all_layer_ids[idx], all_layer_ids[idx + 1])] = []
        self.between_layer_queues = between_layer_queues

        # initialize kv cache
        # TODO: keep entries in KV cache & activation backup that are still useful if new model
        #  and old one share layers
        # TODO: should not get kv-cache size from file, instead, use the current num layers to get
        #  from model_manager (get from file will be incorrect if we may assign different number of
        #  layers to a compute node)
        self.kv_cache = KVCache(layer_ids=sorted(list(self.in_vram_model_layers.keys())),
                                max_capacity=self.kv_cache_capacity)
        self.activation_backup_cache = ActivationBackupCache(layer_ids=sorted(list(self.in_vram_model_layers.keys())),
                                                             max_capacity=self.activation_backup_cache_capacity)

        # calculate available vram
        total_vram_usage: float = 0
        for layer_id in self.in_vram_model_layers:
            total_vram_usage += self.in_vram_model_layers[layer_id].vram_usage
        self.available_vram = self.vram_size - total_vram_usage

    def receive_request(self, request: InferenceRequest) -> None:
        """
        Receives a request from one inbound link after sending is finished.

        :param request: the request to receive
        :return: None
        """
        # check the request's pipeline
        assert request.get_current_pipeline_stage().node_uid == self.node_uid, "Found a mis-routed request!"
        cur_layers_to_infer = sorted(request.get_current_pipeline_stage().layers_to_infer)

        # check inference history
        if len(request.inference_history) == 0:
            last_layer_inferred = -1
        else:
            last_layer_inferred = request.inference_history[-1].layer_id
        assert last_layer_inferred + 1 == min(cur_layers_to_infer), "Inference layer is not continuous!"

        # check the planned inference can happen on current node
        infer_with_current_model: bool = False
        if self.model_status == ModelStatus.NoModel:
            # If there is no model and no plan to load a model (since NoModel means that prepare_loading_model is
            # not called), no request should move this way.
            assert False, f"Receives a request when there is no model on it!"
        elif self.model_status == ModelStatus.Ready:
            # The request will be inferred using the current model. We allow the request to use only a subset of
            # the model for inference.
            for layer_id in cur_layers_to_infer:
                assert layer_id in self.in_vram_model_layers, f"Found incompatible inference request!"
            infer_with_current_model = True
        elif self.model_status == ModelStatus.Flushing:
            if request.request_uid in self.request_uids_to_wait:
                # Requests waited on will use the old model.
                for layer_id in cur_layers_to_infer:
                    assert layer_id in self.in_vram_model_layers, f"Found incompatible inference request!"
                infer_with_current_model = True
            else:
                # Requests not waited on will use the new model.
                for layer_id in cur_layers_to_infer:
                    assert layer_id in self.new_model_layers, f"Found incompatible inference request!"
                infer_with_current_model = False
        elif self.model_status == ModelStatus.Loading:
            # if we are loading the model, then all requests that arrive must use the new model
            for layer_id in cur_layers_to_infer:
                assert layer_id in self.new_model_layers, f"Found incompatible inference request!"
            infer_with_current_model = False

        # check kv cache
        if request.phase == RequestPhase.Increment:
            # TODO: if there is a node failure and this node is backup node, we may allow restoring kv cache
            #  from activation backup, but that will add a cost to the inference time
            self.kv_cache.check_kv_cache(layers=cur_layers_to_infer,
                                         query_uid=request.base_query_uid,
                                         num_prev_tokens=request.prev_num_tokens)

        # add the request
        if infer_with_current_model:
            next_layer_to_infer = min(cur_layers_to_infer)
            if next_layer_to_infer == min(self.in_vram_model_layers.keys()):
                # inference will start from the first layer
                self.inbound_request_queue.append(request)
            else:
                # skip some layers and put the request to the queue before next_layer_to_infer
                self.between_layer_queues[(next_layer_to_infer - 1, next_layer_to_infer)].append(request)
            self.cpu_buffer[request.request_uid] = request
        else:
            # TODO: if the request arrives and does not use current model, we need to cache it and
            #  when the new model is loaded, we can put the requests in corresponding queues
            raise NotImplementedError

    def receive_backup(self, request: InferenceRequest) -> None:
        """
        Receive activation backup of a request.

        :param request: the request for activation backup
        :return: None
        """
        # get layer id to backup
        if len(request.inference_history) == 0:
            assert False, "Trying to make an activation backup for first layer, which is meaningless!"
        else:
            backup_layer_id = request.inference_history[-1].layer_id + 1
        assert backup_layer_id in self.in_vram_model_layers.keys(), "Bad backup layer id!"

        # make sure the backup does not happen on the node the query is executed
        assert not self.kv_cache.exist_duplicate(query_uid=request.base_query_uid, layer_id=backup_layer_id), \
            "Trying to backup a query on the node it is executed!"

        # initialize / grow backup
        if request.phase == RequestPhase.Initialization:
            # initialize activation backup on node & in tracker
            self.activation_backup_cache.initialize_query_backup(layer_id=backup_layer_id,
                                                                 query_uid=request.base_query_uid,
                                                                 num_tokens=request.token_seq_length)
            request.kv_tracker_ref.add_activation_backup_location(layer_id=backup_layer_id,
                                                                  backup_location=self.node_uid)
        elif request.phase == RequestPhase.Increment:
            # grow the existing backup by one
            self.activation_backup_cache.check_activation_backup(layer_id=backup_layer_id,
                                                                 query_uid=request.base_query_uid,
                                                                 num_prev_tokens=request.prev_num_tokens)
            self.activation_backup_cache.grow_query_backup(layer_id=backup_layer_id,
                                                           query_uid=request.base_query_uid)
        else:
            assert False, "Unknown request phase!"

    def is_node_busy(self) -> bool:
        """
        Check whether this node is busy (a batch is in execution on this node).

        :return: whether this node is busy
        """
        return self.current_inference_batch is not None

    def get_current_inference_layer(self) -> int:
        """
        Get current inference layer on the node.

        :return: current inference layer
        """
        return self.current_layer_id

    def get_executable_requests(self) -> List[InferenceRequest]:
        """
        Get all executable requests based on current model status.

        :return: the list of executables requests
        """
        if self.model_status == ModelStatus.Ready or self.model_status == ModelStatus.Flushing:
            # in Ready / Flushing, all requests in the queues are executable
            # return the queue for current layer (model inference happens in a cyclic way)
            if self.current_layer_id == min(self.in_vram_model_layers.keys()):
                return self.inbound_request_queue
            else:
                return self.between_layer_queues[(self.current_layer_id - 1, self.current_layer_id)]
        elif self.model_status == ModelStatus.Loading:
            # loading: no requests are executable
            return []
        elif self.model_status == ModelStatus.NoModel:
            # no model: we should not enter this case
            assert False, "Try to get executable requests when there is no model on node!"
        else:
            assert False, "Unknown model status!"

    def get_inference_statistics(self, requests: List[InferenceRequest], layer_id: int) -> (float, float):
        """
        Get inference statistics for a batch of requests on current layer.

        :param requests: a list of inference requests
        :param layer_id: id of the layer
        :return: (inference_time, inference_vram_usage)
        """
        cur_layer = self.in_vram_model_layers[layer_id]
        cur_layer_time, cur_layer_vram_usage = cur_layer.get_inference_statistics(requests=requests)

        # overhead modeling
        if layer_id == min(self.in_vram_model_layers.keys()):
            num_tokens = sum([req.token_seq_length for req in self.cpu_buffer.values()])
            activation_size = 0 if len(self.cpu_buffer) == 0 else list(self.cpu_buffer.values())[0].activation_size
            concat_overhead = (num_tokens * activation_size) / (4 * gbps)
            transfer_overhead = (num_tokens * activation_size) / (5 * gbps)
            cur_layer_time += concat_overhead + transfer_overhead
            self.cpu_buffer.clear()

        return cur_layer_time, cur_layer_vram_usage

    def get_typical_token_throughput(self) -> float:
        """
        Get typical token throughput of this node.

        :return: typical token throughput
        """
        # get typical inference settings
        assert isinstance(self.inference_settings, InferenceSettings), "Inference settings not found!"
        prompt_typical_requests = self.inference_settings.prompt_typical_requests
        prompt_typical_tokens = self.inference_settings.prompt_typical_tokens
        decode_typical_tokens = self.inference_settings.decode_typical_tokens

        # calculation is dependent on prompt typical requests
        if prompt_typical_requests >= 1:
            # since we are in the linear region, we do not need scaling
            total_time = 0
            total_processed_tokens = prompt_typical_tokens + decode_typical_tokens
            for _, layer in self.in_vram_model_layers.items():
                total_time += layer.get_prompt_inference_time(prompt_phase_tokens=prompt_typical_tokens)
                total_time += layer.get_decode_inference_time(decode_phase_tokens=decode_typical_tokens)
            return total_processed_tokens / total_time
        else:
            # need to scale prompt_typical_requests to 1
            rescaling = 1 / prompt_typical_requests
            rescaled_prompt_tokens = int(prompt_typical_tokens * rescaling)

            total_time = 0
            total_processed_tokens = rescaling * (prompt_typical_tokens + decode_typical_tokens)
            for _, layer in self.in_vram_model_layers.items():
                total_time += layer.get_prompt_inference_time(prompt_phase_tokens=rescaled_prompt_tokens)
                total_time += layer.get_decode_inference_time(decode_phase_tokens=decode_typical_tokens) * rescaling
            return total_processed_tokens / total_time

    def start_execution(self, requests: List[InferenceRequest]) -> InferenceBatch:
        """
        Start execution of a batch of requests.

        :param requests: the list of requests to be inferred
        :return: an InferenceBatch
        """
        # check whether we can do inference at this time
        assert self.model_status == ModelStatus.Ready or self.model_status == ModelStatus.Flushing, "Bad model status!"
        assert self.current_inference_batch is None, "Another batch is already in execution!"

        # check that for all these requests, the current layer is their next layer
        for request in requests:
            if len(request.inference_history) == 0:
                last_layer_inferred = -1
            else:
                last_layer_inferred = request.inference_history[-1].layer_id
            assert last_layer_inferred + 1 == self.current_layer_id, f"Found incompatible inference request!"

        # check that this batch of requests does not violate inference settings
        _prompt_num_request, _prompt_num_tokens, _decode_context, _decode_num_tokens = 0, 0, 0, 0
        for request in requests:
            if request.phase == RequestPhase.Initialization:
                _prompt_num_request += 1
                _prompt_num_tokens += request.token_seq_length
            elif request.phase == RequestPhase.Increment:
                _decode_context += request.prev_num_tokens
                _decode_num_tokens += 1
        assert isinstance(self.inference_settings, InferenceSettings), "No inference settings found!"
        assert _prompt_num_request <= self.inference_settings.prompt_max_requests, "Inference setting violation!"
        assert _prompt_num_tokens <= self.inference_settings.prompt_max_tokens, "Inference setting violation!"
        assert _decode_context <= self.inference_settings.decode_max_context, "Inference setting violation!"
        assert _decode_num_tokens <= self.inference_settings.decode_max_tokens, "Inference setting violation!"

        # get inference statistics
        inference_time, inference_vram_usage = self.get_inference_statistics(
            requests=requests, layer_id=self.current_layer_id
        )
        assert self.available_vram >= inference_vram_usage, "VRAM is not enough for inference!"

        # get the queue we are currently working on
        if self.current_layer_id == min(self.in_vram_model_layers.keys()):
            current_queue = self.inbound_request_queue
        else:
            current_queue = self.between_layer_queues[(self.current_layer_id - 1, self.current_layer_id)]

        # check requests exist and remove them from inbound request queue
        # get uids of requests waiting for execution
        execute_request_uids: Set[int] = set()
        for execute_request in requests:
            assert execute_request.request_uid not in execute_request_uids, "Duplicate request found!"
            execute_request_uids.add(execute_request.request_uid)
        # get uids of all requests currently on node
        all_current_request_uids: Set[int] = set()
        for current_request in current_queue:
            assert current_request.request_uid not in all_current_request_uids, "Duplicate request found!"
            all_current_request_uids.add(current_request.request_uid)
        # check whether all these requests exist
        for execute_request_uid in execute_request_uids:
            assert execute_request_uid in all_current_request_uids, "Found unknown request!"
        # remove the requests from inbound request queue
        new_queue: List[InferenceRequest] = []
        for existing_request in current_queue:
            if existing_request.request_uid not in execute_request_uids:
                new_queue.append(existing_request)
        if self.current_layer_id == min(self.in_vram_model_layers.keys()):
            self.inbound_request_queue = new_queue
        else:
            self.between_layer_queues[(self.current_layer_id - 1, self.current_layer_id)] = new_queue

        # start inference
        inference_batch = InferenceBatch(requests=requests,
                                         duration=inference_time,
                                         vram_usage=inference_vram_usage)
        self.current_inference_batch = inference_batch
        self.available_vram -= inference_vram_usage

        # return the inference batch
        return inference_batch

    def march_to_next_layer(self) -> int:
        """
        March to the next layer that needs inference.
        Note: 1. if all layers do not has pending requests, we will not move

        :return: next layer idx
        """
        all_layer_ids: List[int] = sorted(list(self.in_vram_model_layers.keys()))
        cur_layer_id_position = all_layer_ids.index(self.current_layer_id)
        for pos_offset in range(1, len(all_layer_ids)):
            # get new layer id
            new_pos = (cur_layer_id_position + pos_offset) % len(all_layer_ids)
            new_layer_id = all_layer_ids[new_pos]

            # check whether this layer's input is empty
            if new_layer_id == min(all_layer_ids):
                if not len(self.inbound_request_queue) == 0:
                    self.current_layer_id = new_layer_id
                    return new_layer_id
            else:
                if not len(self.between_layer_queues[(new_layer_id - 1, new_layer_id)]) == 0:
                    self.current_layer_id = new_layer_id
                    return new_layer_id
        return self.current_layer_id

    def finish_execution(self, inference_batch_handle: int) -> Tuple[InferenceBatch, bool]:
        """
        Finish execution of a batch of requests.

        :param inference_batch_handle: handle of the inference batch
        :return: the inference batch that finishes, whether we need to trigger network send to next node
        """
        # check whether the inference batch matches with input
        assert inference_batch_handle == self.current_inference_batch.get_handle(), "Inference batch mismatch!"

        # clear current inference batch & vram usage
        current_inference_batch: InferenceBatch = self.current_inference_batch
        self.current_inference_batch = None
        self.available_vram += current_inference_batch.vram_usage
        assert self.available_vram <= self.vram_size, "Bad available vram size!"

        # mark the requests as inferred
        layer = self.in_vram_model_layers[self.current_layer_id]
        layer.mark_inferred(requests=current_inference_batch.requests, node_uid=self.node_uid)

        # if this layer is the last layer, then we need to:
        # (1) update request_uids_to_wait (only when flushing)
        # (2) mark request as inferred
        if self.current_layer_id == max(self.in_vram_model_layers.keys()):
            # if we are flushing, also need to remove the requests from wait list
            if self.model_status == ModelStatus.Flushing:
                for request in current_inference_batch.requests:
                    assert request.request_uid in self.request_uids_to_wait, "Found request not waited on!"
                    self.request_uids_to_wait.remove(request.request_uid)

            # put the request uids in inferred request uids
            for request in current_inference_batch.requests:
                self.inferred_request_uids.add(request.request_uid)

        # update the requests in kv cache
        for request in current_inference_batch.requests:
            # get layer ids
            kv_update_layer_ids: List[int] = [self.current_layer_id]

            # update kv cache based on request phase
            if request.phase == RequestPhase.Initialization:
                self.kv_cache.initialize_query_kv_cache(layers=kv_update_layer_ids,
                                                        query_uid=request.base_query_uid,
                                                        num_tokens=request.token_seq_length)
                for layer_id in kv_update_layer_ids:
                    request.kv_tracker_ref.add_kv_cache_location(layer_id=layer_id,
                                                                 kv_location=self.node_uid)
            elif request.phase == RequestPhase.Increment:
                self.kv_cache.grow_query_kv_cache(layers=kv_update_layer_ids, query_uid=request.base_query_uid)

        # put the requests into the next queue
        if self.current_layer_id == max(self.in_vram_model_layers.keys()):
            # last layer, put into output dict
            for request in current_inference_batch.requests:
                assert request.request_uid not in self.outbound_request_dict, "Duplicate requests!"
                self.outbound_request_dict[request.request_uid] = request
            trigger_network_send = True
        else:
            # a layer in the middle, put into next queue
            next_queue = self.between_layer_queues[(self.current_layer_id, self.current_layer_id + 1)]
            for request in current_inference_batch.requests:
                assert request.request_uid not in next_queue, "Duplicate requests!"
                next_queue.append(request)
            trigger_network_send = False

        # march to the next layer (otherwise, the next start_execution will continue working on the same layer)
        self.march_to_next_layer()

        # return
        return current_inference_batch, trigger_network_send
