# 2024.04.03 Yixuan Mei

from typing import List

from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec
from simulator.model_manager.base_classes import ModelStatistics
from simulator.model_manager.example_small.example_small_a100 import ExampleSmallA100
from simulator.model_manager.example_small.example_small_t4 import ExampleSmallT4


class ExampleSmallStatistics(ModelStatistics):
    def __init__(self):
        """
        This class stores the profiling results of different machines running ExampleSmall
        """
        # machine profiling results
        self.a100 = ExampleSmallA100()
        self.t4 = ExampleSmallT4()

        # model statistics
        self.token_size: float = 8 * Byte
        self.activation_size: float = 8 * KB
        self.model_param_sizes = [100 * MB, 100 * MB, 100 * MB]

    # ---------------------------------------- Machine ---------------------------------------- #

    def get_profiling_results(self, machine_type: str) -> MachineProfile:
        """
        Get the profiling results of running one layer of the model on given type of machine.

        :param machine_type: machine type
        :return: MachineProfile
        """
        if machine_type == "A100":
            return self.a100.get_profiling_results()
        elif machine_type == "T4":
            return self.t4.get_profiling_results()
        else:
            assert False, "Found unknown machine type"

    def get_max_num_layers(self, machine_type: str) -> int:
        """
        Get the max number of layers the given type of machine can hold.

        :param machine_type: machine type
        :return: max number of layers the given type of machine can hold
        """
        if machine_type == "A100":
            return self.a100.get_max_num_layers()
        elif machine_type == "T4":
            return self.t4.get_max_num_layers()
        else:
            assert False, "Found unknown machine type"

    def get_inference_settings(self, machine_type: str, num_on_node_layers: int) -> InferenceSettings:
        """
        Get the inference settings of the given machine type when there are given number of layers.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: InferenceSettings
        """
        if machine_type == "A100":
            return self.a100.get_inference_settings(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4":
            return self.t4.get_inference_settings(num_on_node_layers=num_on_node_layers)
        else:
            assert False, "Found unknown machine type"

    def get_typical_token_throughput(self, machine_type: str, num_on_node_layers: int) -> float:
        """
        Get the typical token throughput of given machine type when there are given number of layers on node.
        Note: this value is the time needed to infer all these layers

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: typical token throughput (in #tokens / second)
        """
        if machine_type == "A100":
            return self.a100.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4":
            return self.t4.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        else:
            assert False, "Found unknown machine type"

    def get_kv_cache_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the kv cache capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: kv cache capacity
        """
        if machine_type == "A100":
            return self.a100.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4":
            return self.t4.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        else:
            assert False, "Found unknown machine type"

    def get_activation_backup_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the activation backup capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: activation backup capacity
        """
        if machine_type == "A100":
            return self.a100.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4":
            return self.t4.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        else:
            assert False, "Found unknown machine type"

    # ----------------------------------------- Model ----------------------------------------- #

    def get_model_params(self) -> List[float]:
        """
        Get the param size list of the model.

        :return: a list of floats, representing the param size of each layer
        """
        return self.model_param_sizes

    def get_model_token_size(self) -> float:
        """
        Get the token size of the model.

        :return: token size
        """
        return self.token_size

    def get_model_activation_size(self) -> float:
        """
        Get the activation size of the model.

        :return: activation size of a token
        """
        return self.activation_size
