# 2024.04.03 Yixuan Mei

from typing import Dict, List
from abc import ABC, abstractmethod

from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings


class ModelOnMachine(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """
        Base class for storing the profiling results of a model on a machine.
        """
        # profiling results (for one single layer)
        self.prompt_bs2time: Dict[int, float]
        self.prompt_bs2vram: Dict[int, float]
        self.decode_bs2time: Dict[int, float]
        self.decode_bs2vram: Dict[int, float]

        # inference settings
        self.max_num_layers: int
        self.num_layers_to_inference_settings: Dict[int, InferenceSettings]

        # kv cache and activation backup cache
        self.kv_cache_capacity: Dict[int, int]
        self.activation_backup_capacity: Dict[int, int]

    @abstractmethod
    def get_profiling_results(self) -> MachineProfile:
        """
        Get the profiling results of running one layer of the model on the machine.

        :return: MachineProfile
        """
        pass

    @abstractmethod
    def get_max_num_layers(self) -> int:
        """
        Get the max number of layers that can be loaded into this machine.

        :return: max number of layers that can be loaded into the machine
        """
        pass

    @abstractmethod
    def get_inference_settings(self, num_on_node_layers: int) -> InferenceSettings:
        """
        Get the inference settings when there are given number of layers on node.
        Note: The inference settings should be dependent on the number of layers.

        :param num_on_node_layers: number of layers on node
        :return: inference settings
        """
        pass

    @abstractmethod
    def get_typical_token_throughput(self, num_on_node_layers: int) -> float:
        """
        Get typical token throughput when there are given number of layers on node.

        :param num_on_node_layers: number of layers on node
        :return: typical token throughput (in #tokens/s)
        """
        pass

    @abstractmethod
    def get_kv_cache_capacity(self, num_on_node_layers: int) -> int:
        """
        Get the kv cache capacity of this machine when using the current model.

        :param num_on_node_layers: number of layers on node
        :return: kv cache capacity
        """
        pass

    @abstractmethod
    def get_activation_backup_capacity(self, num_on_node_layers: int) -> int:
        """
        Get the activation backup capacity of this machine when using the current model.

        :param num_on_node_layers: number of layers on node
        :return: activation backup capacity
        """
        pass


class ModelStatistics(ABC):
    # ---------------------------------------- Machine ---------------------------------------- #
    @abstractmethod
    def get_profiling_results(self, machine_type: str) -> MachineProfile:
        """
        Get the profiling results of running one layer of the model on given type of machine.

        :param machine_type: machine type
        :return: MachineProfile
        """
        pass

    @abstractmethod
    def get_max_num_layers(self, machine_type: str) -> int:
        """
        Get the max number of layers the given type of machine can hold.

        :param machine_type: machine type
        :return: max number of layers the given type of machine can hold
        """
        pass

    @abstractmethod
    def get_inference_settings(self, machine_type: str, num_on_node_layers: int) -> InferenceSettings:
        """
        Get the inference settings of the given machine type when there are given number of layers.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: InferenceSettings
        """
        pass

    @abstractmethod
    def get_typical_token_throughput(self, machine_type: str, num_on_node_layers: int) -> float:
        """
        Get the typical token throughput of given machine type when there are given number of layers on node.
        Note: this value considers the time needed to infer all these layers

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: typical token throughput (in #tokens / second)
        """
        pass

    @abstractmethod
    def get_kv_cache_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the kv cache capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: kv cache capacity
        """
        pass

    @abstractmethod
    def get_activation_backup_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the activation backup capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: activation backup capacity
        """
        pass

    # ----------------------------------------- Model ----------------------------------------- #

    @abstractmethod
    def get_model_params(self) -> List[float]:
        """
        Get the param size list of the model.

        :return: a list of floats, representing the param size of each layer
        """
        pass

    @abstractmethod
    def get_model_token_size(self) -> float:
        """
        Get the token size of the model.

        :return: token size
        """
        pass

    @abstractmethod
    def get_model_activation_size(self) -> float:
        """
        Get the activation size of the model.

        :return: activation size of a token
        """
        pass
