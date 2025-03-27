# 2024.04.04 Yixuan Mei

from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec


class XXXModelYYY(ModelOnMachine):
    def __init__(self) -> None:
        """
        # TODO
        This class stores the profiling results of running XXX Model on YYY
        """
        # profiling results (for one single layer)
        self.prompt_bs2time: Dict[int, float] = {
            0: 0,  # TODO
        }
        self.prompt_bs2vram: Dict[int, float] = {
            0: 0,  # TODO
        }
        self.decode_bs2time: Dict[int, float] = {
            0: 0,  # TODO
        }
        self.decode_bs2vram: Dict[int, float] = {
            0: 0,  # TODO
        }

        # inference settings
        self.max_num_layers: int = -1  # TODO
        self.num_layers_to_inference_settings: Dict[int, InferenceSettings] = {
            # TODO
        }

        # kv cache and activation backup cache
        self.kv_cache_capacity: Dict[int, int] = {}  # TODO
        self.activation_backup_capacity: Dict[int, int] = {}  # TODO

    def get_profiling_results(self) -> MachineProfile:
        """
        Get the profiling results of running one layer of the model on the machine.

        :return: MachineProfile
        """
        machine_profile = MachineProfile(prompt_bs2time=self.prompt_bs2time, prompt_bs2vram=self.prompt_bs2vram,
                                         decode_bs2time=self.decode_bs2time, decode_bs2vram=self.decode_bs2vram)
        return machine_profile

    def get_max_num_layers(self) -> int:
        """
        Get the max number of layers that can be loaded into this machine.

        :return: max number of layers that can be loaded into the machine
        """
        return self.max_num_layers

    def get_inference_settings(self, num_on_node_layers: int) -> InferenceSettings:
        """
        Get the inference settings when there are given number of layers on node.
        Note: The inference settings are dependent on the number of layers.

        :param num_on_node_layers: number of layers on node
        :return: inference settings
        """
        assert 0 < num_on_node_layers <= self.max_num_layers, "Bad number of layers on node!"
        return self.num_layers_to_inference_settings[num_on_node_layers]

    def get_typical_token_throughput(self, num_on_node_layers: int) -> float:
        """
        Get typical token throughput when there are given number of layers on node.

        :param num_on_node_layers: number of layers on node
        :return: typical token throughput (in #tokens/s)
        """
        inference_settings = self.get_inference_settings(num_on_node_layers=num_on_node_layers)
        prompt_typical_requests = inference_settings.prompt_typical_requests
        prompt_typical_tokens = inference_settings.prompt_typical_tokens
        decode_typical_tokens = inference_settings.decode_typical_tokens

        # some helper functions
        from simulator.event_simulator.utils import linear_interpolate

        def _get_prompt_time(prompt_num_tokens: int) -> float:
            prompt_left, prompt_right = -1, 1000 * 1000
            for prompt_point in self.prompt_bs2time:
                if prompt_left < prompt_point <= prompt_num_tokens:
                    prompt_left = prompt_point
                if prompt_num_tokens <= prompt_point < prompt_right:
                    prompt_right = prompt_point
            return linear_interpolate(x_0=prompt_left, y_0=self.prompt_bs2time[prompt_left],
                                      x_1=prompt_right, y_1=self.prompt_bs2time[prompt_right],
                                      x_target=prompt_num_tokens)

        def _get_decode_time(decode_num_tokens: int) -> float:
            decode_left, decode_right = -1, 1000 * 1000
            for decode_point in self.decode_bs2time:
                if decode_left < decode_point <= decode_num_tokens:
                    decode_left = decode_point
                if decode_num_tokens <= decode_point < decode_right:
                    decode_right = decode_point
            return linear_interpolate(x_0=decode_left, y_0=self.decode_bs2time[decode_left],
                                      x_1=decode_right, y_1=self.decode_bs2time[decode_right],
                                      x_target=decode_num_tokens)

        # calculation method is dependent on prompt typical requests
        if prompt_typical_requests >= 1:
            # in linear region, no need to rescale
            total_tokens = prompt_typical_tokens + decode_typical_tokens
            layer_prompt_time = _get_prompt_time(prompt_num_tokens=prompt_typical_tokens)
            layer_decode_time = _get_decode_time(decode_num_tokens=decode_typical_tokens)
            total_time = num_on_node_layers * (layer_prompt_time + layer_decode_time)
            return total_tokens / total_time
        else:
            # need to scale to 1
            rescaling = 1 / prompt_typical_requests
            total_tokens = rescaling * (prompt_typical_tokens + decode_typical_tokens)
            layer_prompt_time = _get_prompt_time(prompt_num_tokens=int(prompt_typical_tokens * rescaling))
            layer_decode_time = _get_decode_time(decode_num_tokens=decode_typical_tokens) * rescaling
            total_time = num_on_node_layers * (layer_prompt_time + layer_decode_time)
            return total_tokens / total_time

    def get_kv_cache_capacity(self, num_on_node_layers: int) -> int:
        """
        Get the kv cache capacity of this machine when using the current model.

        :param num_on_node_layers: number of layers on node
        :return: kv cache capacity
        """
        return self.kv_cache_capacity[num_on_node_layers]

    def get_activation_backup_capacity(self, num_on_node_layers: int) -> int:
        """
        Get the activation backup capacity of this machine when using the current model.

        :param num_on_node_layers: number of layers on node
        :return: activation backup capacity
        """
        return self.activation_backup_capacity[num_on_node_layers]
