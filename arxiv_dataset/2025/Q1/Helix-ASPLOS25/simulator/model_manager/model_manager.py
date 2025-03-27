# 2024.04.03 Yixuan Mei

from enum import Enum
from typing import List, Dict

from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.model_manager.example_small.example_small import ExampleSmallStatistics
from simulator.model_manager.example_large.example_large import ExampleLargeStatistics
from simulator.model_manager.llama2_70b.llama2_70b import LLaMa70BStatistics
from simulator.model_manager.llama1_30b.llama1_30b import LLaMa30BStatistics


class ModelName(Enum):
    # ---------------- Example Models ---------------- #
    ExampleSmall = "ModelName.ExampleSmall"
    ExampleLarge = "ModelName.ExampleLarge"
    # ------------------ Real Models ----------------- #
    LLaMa70B = "ModelName.LLaMa70B"
    LLaMa30B = "ModelName.LLaMa30B"


class ModelManager:
    def __init__(self, model_name: ModelName, machine_num_dict: Dict[str, int]) -> None:
        """
        Model manager.

        :param model_name: name of the LLM
        :param machine_num_dict: {machine_name -> num of machine}
        :return: None
        """
        # model name
        self.model_name: ModelName = model_name
        self.machine_num_dict: Dict[str, int] = machine_num_dict
        self.allow_force_set: bool = False

        # model
        if self.model_name == ModelName.ExampleSmall:
            self.model_statistics = ExampleSmallStatistics()
        elif self.model_name == ModelName.ExampleLarge:
            self.model_statistics = ExampleLargeStatistics()
        elif self.model_name == ModelName.LLaMa70B:
            self.model_statistics = LLaMa70BStatistics(num_machines_dict=machine_num_dict)
        elif self.model_name == ModelName.LLaMa30B:
            self.model_statistics = LLaMa30BStatistics(num_machines_dict=machine_num_dict)
        else:
            assert False, "Unknown model name!"

    # ---------------------------------------- Machine ---------------------------------------- #

    def get_profiling_results(self, machine_type: str) -> MachineProfile:
        """
        Get the profiling results of running one layer of the model on given type of machine.

        :param machine_type: machine type
        :return: machine profile
        """
        return self.model_statistics.get_profiling_results(machine_type=machine_type)

    def get_max_num_layers(self, machine_type: str) -> int:
        """
        Get the max number of layers the given type of machine can hold.

        :param machine_type: machine type
        :return: max number of layers the given type of machine can hold
        """
        return self.model_statistics.get_max_num_layers(machine_type=machine_type)

    def get_inference_settings(self, machine_type: str, num_on_node_layers: int) -> InferenceSettings:
        """
        Get the inference settings of the given machine type when there are given number of layers.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: InferenceSettings
        """
        if self.allow_force_set:
            # this is only used when you want to load more than the recommended number of layers
            max_num_layers = self.get_max_num_layers(machine_type=machine_type)
            num_on_node_layers = min(num_on_node_layers, max_num_layers)
        return self.model_statistics.get_inference_settings(machine_type=machine_type,
                                                            num_on_node_layers=num_on_node_layers)

    def get_typical_token_throughput(self, machine_type: str, num_on_node_layers: int) -> float:
        """
        Get the typical token throughput of given machine type when there are given number of layers on node.
        Note: this value considers the time needed to infer all these layers

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: typical token throughput (in #tokens / second)
        """
        return self.model_statistics.get_typical_token_throughput(machine_type=machine_type,
                                                                  num_on_node_layers=num_on_node_layers)

    def get_kv_cache_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the kv cache capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: kv cache capacity
        """
        return self.model_statistics.get_kv_cache_capacity(machine_type=machine_type,
                                                           num_on_node_layers=num_on_node_layers)

    def get_activation_backup_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the activation backup capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: activation backup capacity
        """
        return self.model_statistics.get_activation_backup_capacity(machine_type=machine_type,
                                                                    num_on_node_layers=num_on_node_layers)

    # ----------------------------------------- Model ----------------------------------------- #

    def get_model_params(self) -> List[float]:
        """
        Get the param size list of the model.

        :return: a list of floats, representing the param size of each layer
        """
        return self.model_statistics.get_model_params()

    def get_num_layers(self) -> int:
        """
        Get the number of layers of the model.

        :return: number of layers
        """
        return len(self.model_statistics.get_model_params())

    def get_model_token_size(self) -> float:
        """
        Get the token size of the model.

        :return: token size
        """
        return self.model_statistics.get_model_token_size()

    def get_model_activation_size(self) -> float:
        """
        Get the activation size of the model.

        :return: activation size of a token
        """
        return self.model_statistics.get_model_activation_size()
