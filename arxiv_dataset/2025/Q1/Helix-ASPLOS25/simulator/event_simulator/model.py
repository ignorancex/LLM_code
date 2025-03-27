# 2023.12.11 Yixuan Mei

from typing import Dict, List
from enum import Enum

from simulator.event_simulator.request import InferenceRequest, RequestPhase
from simulator.event_simulator.utils import linear_interpolate


class ModelStatus(Enum):
    """ Status of the model on GPU compute nodes """
    NoModel = "ModelStatus.NoModel"
    Flushing = "ModelStatus.Flushing"
    Loading = "ModelStatus.Loading"
    Ready = "ModelStatus.Ready"


class MachineProfile:
    def __init__(self, prompt_bs2time: Dict[int, float], prompt_bs2vram: Dict[int, float],
                 decode_bs2time: Dict[int, float], decode_bs2vram: Dict[int, float]) -> None:
        """
        Machine Profile. The values are only dependent on:
            1. the GPU on machine
            2. the LLM model
        We assume that all layers are the same in the LLM.

        :param prompt_bs2time: batch size -> running time (prompt phase)
        :param prompt_bs2vram: batch size -> vram usage   (prompt phase)
        :param decode_bs2time: batch size -> running time (decode phase)
        :param decode_bs2vram: batch size -> vram usage   (decode phase)
        """
        self.prompt_bs2time = prompt_bs2time
        self.prompt_bs2vram = prompt_bs2vram
        self.decode_bs2time = decode_bs2time
        self.decode_bs2vram = decode_bs2vram


class ModelLayer:
    def __init__(self, layer_id: int, vram_usage: float) -> None:
        """
        One Model layer.

        :param layer_id: id of this layer
        :param vram_usage: vram usage of the parameters
        """
        # basic properties
        self.layer_id: int = layer_id
        self.vram_usage: float = vram_usage

        # inference statistics (dependent on the machine used)
        self.prompt_bs2time: Dict[int, float] = {}
        self.prompt_bs2vram: Dict[int, float] = {}
        self.decode_bs2time: Dict[int, float] = {}
        self.decode_bs2vram: Dict[int, float] = {}

    def set_layer_statistics(self, machine_profile: MachineProfile) -> None:
        """
        Set run time & vram usage statistics for running the model on this machine.

        :param machine_profile: profiling results of the LLM on current machine
        :return: None
        """
        self.prompt_bs2time = machine_profile.prompt_bs2time
        self.prompt_bs2vram = machine_profile.prompt_bs2vram
        self.decode_bs2time = machine_profile.decode_bs2time
        self.decode_bs2vram = machine_profile.decode_bs2vram
        assert sorted(list(self.prompt_bs2time.keys())) == sorted(list(self.prompt_bs2vram.keys())), \
            "Keys of profiled data mismatch in prompt phase!"
        assert sorted(list(self.decode_bs2time.keys())) == sorted(list(self.decode_bs2vram.keys())), \
            "Keys of profiled data mismatch in decode phase!"

    def get_inference_statistics(self, requests: List[InferenceRequest]) -> (float, float):
        """
        Get inference time & vram usage for given request.
        Notes:
        1. Our performance modeling is based on: https://arxiv.org/pdf/2305.02440.pdf. Note that we need to
           consider prompt phase and decode phase separately, since their memory access pattern significantly
           affects their running time.
        2. The run time w.r.t. batch size should be: first a constant, then linear scaling. In our case, we
           are already in the linear zone even with the smallest batch size.
        3. We interpolate the profiling data to get the statistics of the real batch.

        :param requests: a list of inference requests
        :return: (inference_time, inference_vram_usage)
        """
        # count number of tokens to process in each phase
        prompt_phase_tokens, decode_phase_tokens = 0, 0
        for request in requests:
            if request.phase == RequestPhase.Initialization:
                prompt_phase_tokens += request.token_seq_length
            elif request.phase == RequestPhase.Increment:
                assert request.token_seq_length == 1, "In decode phase token sequence length must be 1!"
                decode_phase_tokens += request.token_seq_length
            else:
                assert False, "Found unknown reqeust phase!"

        # get the two end-points of interpolation for prompt phase
        prompt_left, prompt_right = -1, 10000 * 10000
        for prompt_point in self.prompt_bs2time.keys():
            if prompt_left < prompt_point <= prompt_phase_tokens:
                prompt_left = prompt_point
            if prompt_phase_tokens <= prompt_point < prompt_right:
                prompt_right = prompt_point
        assert prompt_left in self.prompt_bs2time and prompt_right in self.prompt_bs2time, \
            f"Can not interpolate for bs={prompt_phase_tokens} in prompt phase!"

        # get the two end-points of interpolation for decode phase
        decode_left, decode_right = -1, 10000 * 10000
        for decode_point in self.decode_bs2time.keys():
            if decode_left < decode_point <= decode_phase_tokens:
                decode_left = decode_point
            if decode_phase_tokens <= decode_point < decode_right:
                decode_right = decode_point
        assert decode_left in self.decode_bs2time and decode_right in self.decode_bs2time, \
            f"Can not interpolate for bs={decode_phase_tokens} in decode phase!"

        # perform linear interpolation
        prompt_time = linear_interpolate(x_0=prompt_left, y_0=self.prompt_bs2time[prompt_left],
                                         x_1=prompt_right, y_1=self.prompt_bs2time[prompt_right],
                                         x_target=prompt_phase_tokens)
        prompt_vram = linear_interpolate(x_0=prompt_left, y_0=self.prompt_bs2vram[prompt_left],
                                         x_1=prompt_right, y_1=self.prompt_bs2vram[prompt_right],
                                         x_target=prompt_phase_tokens)
        decode_time = linear_interpolate(x_0=decode_left, y_0=self.decode_bs2time[decode_left],
                                         x_1=decode_right, y_1=self.decode_bs2time[decode_right],
                                         x_target=decode_phase_tokens)
        decode_vram = linear_interpolate(x_0=decode_left, y_0=self.decode_bs2vram[decode_left],
                                         x_1=decode_right, y_1=self.decode_bs2vram[decode_right],
                                         x_target=decode_phase_tokens)

        if decode_phase_tokens == 1:
            decode_time = decode_time * 2

        return prompt_time + decode_time, prompt_vram + decode_vram

    def get_prompt_inference_time(self, prompt_phase_tokens: int) -> float:
        """
        Get prompt inference time for a batch of num_tokens. (Used by ComputeNode to compute typical throughput)

        :param prompt_phase_tokens: number of tokens in the batch
        :return: inference time on this layer
        """
        # get the two end-points of interpolation for prompt phase
        prompt_left, prompt_right = -1, 10000 * 10000
        for prompt_point in self.prompt_bs2time.keys():
            if prompt_left < prompt_point <= prompt_phase_tokens:
                prompt_left = prompt_point
            if prompt_phase_tokens <= prompt_point < prompt_right:
                prompt_right = prompt_point
        assert prompt_left in self.prompt_bs2time and prompt_right in self.prompt_bs2time, \
            f"Can not interpolate for bs={prompt_phase_tokens} in prompt phase!"

        # get prompt phase time
        prompt_time = linear_interpolate(x_0=prompt_left, y_0=self.prompt_bs2time[prompt_left],
                                         x_1=prompt_right, y_1=self.prompt_bs2time[prompt_right],
                                         x_target=prompt_phase_tokens)
        return prompt_time

    def get_decode_inference_time(self, decode_phase_tokens: int) -> float:
        """
        Get decode inference time for a batch of num_tokens. (Used by ComputeNode to compute typical throughput)

        :param decode_phase_tokens: number of tokens in the batch
        :return: inference time on this layer
        """
        # get the two end-points of interpolation for decode phase
        decode_left, decode_right = -1, 10000 * 10000
        for decode_point in self.decode_bs2time.keys():
            if decode_left < decode_point <= decode_phase_tokens:
                decode_left = decode_point
            if decode_phase_tokens <= decode_point < decode_right:
                decode_right = decode_point
        assert decode_left in self.decode_bs2time and decode_right in self.decode_bs2time, \
            f"Can not interpolate for bs={decode_phase_tokens} in decode phase!"

        # get decode phase time
        decode_time = linear_interpolate(x_0=decode_left, y_0=self.decode_bs2time[decode_left],
                                         x_1=decode_right, y_1=self.decode_bs2time[decode_right],
                                         x_target=decode_phase_tokens)
        return decode_time

    def mark_inferred(self, requests: List[InferenceRequest], node_uid: int) -> None:
        """
        Infer request using this layer.

        :param requests: a list of inference requests
        :param node_uid: which node is this model layer on
        :return: None
        """
        for request in requests:
            request.add_inference_history(layers=[self.layer_id], node_uid=node_uid)


def create_model(layer_parameter_sizes: List[float]) -> Dict[int, ModelLayer]:
    """
    Create a model with given parameter size for each layer. Notice that the model does not
    contain inference statistics.

    :param layer_parameter_sizes: each layer's parameter size
    :return: the new model
    """
    model_dict: Dict[int, ModelLayer] = {}
    for layer_idx, parameter_size in enumerate(layer_parameter_sizes):
        model_dict[layer_idx] = ModelLayer(layer_id=layer_idx, vram_usage=parameter_size)
    return model_dict
