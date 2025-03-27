# 2024.04.04 Yixuan Mei

from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec


class ExampleLargeT4(ModelOnMachine):
    def __init__(self) -> None:
        """
        This class stores the profiling results of running ExampleLarge on T4.
        """
        # profiling results (for one single layer)
        self.prompt_bs2time: Dict[int, float] = {
            0: 0, 10: 50 * MilliSec, 20: 51 * MilliSec, 30: 51 * MilliSec, 40: 51 * MilliSec, 50: 70 * MilliSec,
            60: 91 * MilliSec, 70: 111 * MilliSec, 80: 132 * MilliSec, 90: 151 * MilliSec, 100: 183 * MilliSec,
            110: 205 * MilliSec, 120: 227 * MilliSec, 130: 248 * MilliSec, 140: 270 * MilliSec, 150: 292 * MilliSec,
            160: 314 * MilliSec, 170: 336 * MilliSec, 180: 358 * MilliSec, 190: 380 * MilliSec, 200: 402 * MilliSec
        }
        self.prompt_bs2vram: Dict[int, float] = {
            i: i * MB for i in range(0, 201, 10)
        }
        self.decode_bs2time: Dict[int, float] = {
            0: 0, 1: 50 * MilliSec, 2: 51 * MilliSec, 3: 51 * MilliSec, 4: 51 * MilliSec, 5: 70 * MilliSec,
            6: 91 * MilliSec, 7: 111 * MilliSec, 8: 132 * MilliSec, 9: 151 * MilliSec, 10: 183 * MilliSec,
            11: 205 * MilliSec, 12: 227 * MilliSec, 13: 248 * MilliSec, 14: 270 * MilliSec, 15: 292 * MilliSec,
            16: 314 * MilliSec, 17: 336 * MilliSec, 18: 358 * MilliSec, 19: 380 * MilliSec, 20: 402 * MilliSec
        }
        self.decode_bs2vram: Dict[int, float] = {
            i: i * MB for i in range(0, 21, 1)
        }

        # inference settings
        self.max_num_layers: int = 2
        self.num_layers_to_inference_settings: Dict[int, InferenceSettings] = {
            1: InferenceSettings(prompt_max_requests=5,
                                 prompt_max_tokens=200,
                                 prompt_typical_requests=4,
                                 prompt_typical_tokens=160,
                                 decode_max_context=400,
                                 decode_max_tokens=20,
                                 decode_typical_tokens=16),
            2: InferenceSettings(prompt_max_requests=5,
                                 prompt_max_tokens=200,
                                 prompt_typical_requests=4,
                                 prompt_typical_tokens=160,
                                 decode_max_context=400,
                                 decode_max_tokens=20,
                                 decode_typical_tokens=16),
        }

        # kv cache and activation backup cache
        self.kv_cache_capacity: Dict[int, int] = {1: 51200, 2: 25600}
        self.activation_backup_capacity: Dict[int, int] = {1: 51200, 2: 25600}

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
