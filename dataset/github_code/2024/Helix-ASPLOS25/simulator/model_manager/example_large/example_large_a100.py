# 2024.04.04 Yixuan Mei

from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec


class ExampleLargeA100(ModelOnMachine):
    def __init__(self) -> None:
        """
        This class stores the profiling results of running ExampleLarge on A100.
        """
        # profiling results (for one single layer)
        self.prompt_bs2time: Dict[int, float] = {
            0: 0, 10: 20 * MilliSec, 20: 21 * MilliSec, 30: 21 * MilliSec, 40: 21 * MilliSec, 50: 22 * MilliSec,
            60: 22 * MilliSec, 70: 23 * MilliSec, 80: 24 * MilliSec, 90: 35 * MilliSec, 100: 45 * MilliSec,
            110: 56 * MilliSec, 120: 65 * MilliSec, 130: 76 * MilliSec, 140: 86 * MilliSec, 150: 97 * MilliSec,
            160: 107 * MilliSec, 170: 118 * MilliSec, 180: 128 * MilliSec, 190: 139 * MilliSec, 200: 149 * MilliSec,
            210: 160 * MilliSec, 220: 170 * MilliSec, 230: 181 * MilliSec, 240: 191 * MilliSec, 250: 202 * MilliSec,
            260: 212 * MilliSec, 270: 223 * MilliSec, 280: 233 * MilliSec, 290: 244 * MilliSec, 300: 254 * MilliSec,
            310: 265 * MilliSec, 320: 275 * MilliSec
        }
        self.prompt_bs2vram: Dict[int, float] = {
            i: i * MB for i in range(0, 321, 10)
        }
        self.decode_bs2time: Dict[int, float] = {
            0: 0, 1: 20 * MilliSec, 2: 21 * MilliSec, 3: 21 * MilliSec, 4: 21 * MilliSec, 5: 22 * MilliSec,
            6: 22 * MilliSec, 7: 23 * MilliSec, 8: 24 * MilliSec, 9: 35 * MilliSec, 10: 45 * MilliSec,
            11: 56 * MilliSec, 12: 65 * MilliSec, 13: 76 * MilliSec, 14: 86 * MilliSec, 15: 97 * MilliSec,
            16: 107 * MilliSec, 17: 118 * MilliSec, 18: 128 * MilliSec, 19: 139 * MilliSec, 20: 149 * MilliSec,
            21: 160 * MilliSec, 22: 170 * MilliSec, 23: 181 * MilliSec, 24: 191 * MilliSec, 25: 202 * MilliSec,
            26: 212 * MilliSec, 27: 223 * MilliSec, 28: 233 * MilliSec, 29: 244 * MilliSec, 30: 254 * MilliSec,
            31: 265 * MilliSec, 32: 275 * MilliSec
        }
        self.decode_bs2vram: Dict[int, float] = {
            i: i * MB for i in range(0, 33, 1)
        }

        # inference settings
        self.max_num_layers: int = 4
        self.num_layers_to_inference_settings: Dict[int, InferenceSettings] = {
            1: InferenceSettings(prompt_max_requests=5,
                                 prompt_max_tokens=320,
                                 prompt_typical_requests=4,
                                 prompt_typical_tokens=240,
                                 decode_max_context=400,
                                 decode_max_tokens=32,
                                 decode_typical_tokens=24),
            2: InferenceSettings(prompt_max_requests=5,
                                 prompt_max_tokens=320,
                                 prompt_typical_requests=4,
                                 prompt_typical_tokens=220,
                                 decode_max_context=400,
                                 decode_max_tokens=32,
                                 decode_typical_tokens=22),
            3: InferenceSettings(prompt_max_requests=5,
                                 prompt_max_tokens=320,
                                 prompt_typical_requests=4,
                                 prompt_typical_tokens=200,
                                 decode_max_context=400,
                                 decode_max_tokens=32,
                                 decode_typical_tokens=20),
            4: InferenceSettings(prompt_max_requests=5,
                                 prompt_max_tokens=320,
                                 prompt_typical_requests=4,
                                 prompt_typical_tokens=180,
                                 decode_max_context=400,
                                 decode_max_tokens=32,
                                 decode_typical_tokens=18),
        }

        # kv cache and activation backup cache
        self.kv_cache_capacity: Dict[int, int] = {1: 102400, 2: 51200, 3: 38400, 4: 25600}
        self.activation_backup_capacity: Dict[int, int] = {1: 102400, 2: 51200, 3: 38400, 4: 25600}

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
