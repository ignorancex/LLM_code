# 2024.04.04 Yixuan Mei

from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec


class ExampleLargeH100(ModelOnMachine):
    def __init__(self) -> None:
        """
        This class stores the profiling results of running ExampleLarge on H100.
        """
        # profiling results (for one single layer)
        self.prompt_bs2time: Dict[int, float] = {
            0: 0, 10: 15 * MilliSec, 20: 15 * MilliSec, 30: 15 * MilliSec, 40: 16 * MilliSec, 50: 16 * MilliSec,
            60: 16 * MilliSec, 70: 16 * MilliSec, 80: 17 * MilliSec, 90: 17 * MilliSec, 100: 17 * MilliSec,
            110: 17 * MilliSec, 120: 17 * MilliSec, 130: 23 * MilliSec, 140: 28 * MilliSec, 150: 33 * MilliSec,
            160: 38 * MilliSec, 170: 42 * MilliSec, 180: 46 * MilliSec, 190: 51 * MilliSec, 200: 57 * MilliSec,
            210: 62 * MilliSec, 220: 67 * MilliSec, 230: 72 * MilliSec, 240: 77 * MilliSec, 250: 82 * MilliSec,
            260: 87 * MilliSec, 270: 92 * MilliSec, 280: 97 * MilliSec, 290: 102 * MilliSec, 300: 107 * MilliSec,
            310: 112 * MilliSec, 320: 117 * MilliSec, 330: 122 * MilliSec, 340: 127 * MilliSec, 350: 132 * MilliSec,
            360: 137 * MilliSec, 370: 142 * MilliSec, 380: 147 * MilliSec, 390: 152 * MilliSec, 400: 157 * MilliSec,
            410: 162 * MilliSec, 420: 167 * MilliSec, 430: 172 * MilliSec, 440: 177 * MilliSec, 450: 182 * MilliSec,
            460: 187 * MilliSec, 470: 192 * MilliSec, 480: 197 * MilliSec, 490: 202 * MilliSec, 500: 207 * MilliSec
        }
        self.prompt_bs2vram: Dict[int, float] = {
            i: i * MB for i in range(0, 501, 10)
        }
        self.decode_bs2time: Dict[int, float] = {
            0: 0, 1: 15 * MilliSec, 2: 15 * MilliSec, 3: 15 * MilliSec, 4: 16 * MilliSec, 5: 16 * MilliSec,
            6: 16 * MilliSec, 7: 16 * MilliSec, 8: 17 * MilliSec, 9: 17 * MilliSec, 10: 17 * MilliSec,
            11: 17 * MilliSec, 12: 17 * MilliSec, 13: 23 * MilliSec, 14: 28 * MilliSec, 15: 33 * MilliSec,
            16: 38 * MilliSec, 17: 42 * MilliSec, 18: 46 * MilliSec, 19: 51 * MilliSec, 20: 57 * MilliSec,
            21: 62 * MilliSec, 22: 67 * MilliSec, 23: 72 * MilliSec, 24: 77 * MilliSec, 25: 82 * MilliSec,
            26: 87 * MilliSec, 27: 92 * MilliSec, 28: 97 * MilliSec, 29: 102 * MilliSec, 30: 107 * MilliSec,
            31: 112 * MilliSec, 32: 117 * MilliSec, 33: 122 * MilliSec, 34: 127 * MilliSec, 35: 132 * MilliSec,
            36: 137 * MilliSec, 37: 142 * MilliSec, 38: 147 * MilliSec, 39: 152 * MilliSec, 40: 157 * MilliSec,
            41: 162 * MilliSec, 42: 167 * MilliSec, 43: 172 * MilliSec, 44: 177 * MilliSec, 45: 182 * MilliSec,
            46: 187 * MilliSec, 47: 192 * MilliSec, 48: 197 * MilliSec, 49: 202 * MilliSec, 50: 207 * MilliSec
        }
        self.decode_bs2vram: Dict[int, float] = {
            i: i * MB for i in range(0, 51, 1)
        }

        # inference settings
        self.max_num_layers: int = 6
        self.num_layers_to_inference_settings: Dict[int, InferenceSettings] = {
            1: InferenceSettings(prompt_max_requests=8,
                                 prompt_max_tokens=500,
                                 prompt_typical_requests=6,
                                 prompt_typical_tokens=360,
                                 decode_max_context=500,
                                 decode_max_tokens=50,
                                 decode_typical_tokens=36),
            2: InferenceSettings(prompt_max_requests=8,
                                 prompt_max_tokens=500,
                                 prompt_typical_requests=6,
                                 prompt_typical_tokens=340,
                                 decode_max_context=500,
                                 decode_max_tokens=50,
                                 decode_typical_tokens=34),
            3: InferenceSettings(prompt_max_requests=8,
                                 prompt_max_tokens=500,
                                 prompt_typical_requests=6,
                                 prompt_typical_tokens=320,
                                 decode_max_context=500,
                                 decode_max_tokens=50,
                                 decode_typical_tokens=32),
            4: InferenceSettings(prompt_max_requests=8,
                                 prompt_max_tokens=500,
                                 prompt_typical_requests=6,
                                 prompt_typical_tokens=300,
                                 decode_max_context=500,
                                 decode_max_tokens=50,
                                 decode_typical_tokens=30),
            5: InferenceSettings(prompt_max_requests=8,
                                 prompt_max_tokens=500,
                                 prompt_typical_requests=6,
                                 prompt_typical_tokens=280,
                                 decode_max_context=500,
                                 decode_max_tokens=50,
                                 decode_typical_tokens=28),
            6: InferenceSettings(prompt_max_requests=8,
                                 prompt_max_tokens=500,
                                 prompt_typical_requests=6,
                                 prompt_typical_tokens=260,
                                 decode_max_context=500,
                                 decode_max_tokens=50,
                                 decode_typical_tokens=26),
        }

        # kv cache and activation backup cache
        self.kv_cache_capacity: Dict[int, int] = {1: 204800, 2: 102400, 3: 76800, 4: 51200, 5: 40000, 6: 30000}
        self.activation_backup_capacity: Dict[int, int] = {1: 204800, 2: 102400, 3: 76800, 4: 51200, 5: 40000, 6: 30000}

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
