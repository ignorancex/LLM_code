# 2024.04.11 Yixuan Mei

from typing import List, Dict

from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.model_manager.base_classes import ModelStatistics
from simulator.model_manager.llama1_30b.t4.llama1_30b_t4 import LLaMa30BonT4
from simulator.model_manager.llama1_30b.t4x2.llama1_30b_t4x2 import LLaMa30BonT4x2
from simulator.model_manager.llama1_30b.t4x4.llama1_30b_t4x4 import LLaMa30BonT4x4
from simulator.model_manager.llama1_30b.l4.llama1_30b_l4 import LLaMa30BonL4
from simulator.model_manager.llama1_30b.l4x2.llama1_30b_l4x2 import LLaMa30BonL4x2
from simulator.model_manager.llama1_30b.v100.llama1_30b_v100 import LLaMa30BonV100
from simulator.model_manager.llama1_30b.a100.llama1_30b_a100 import LLaMa30BonA100
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec, LLaMa1_30B_TOTAL_LAYERS


class LLaMa30BStatistics(ModelStatistics):
    def __init__(self, num_machines_dict: Dict[str, int]):
        """
        This class stores the profiling results of different machines running LLaMa30B.
        num_machines_dict is a subset of {"A100": 0, "V100": 0, "L4": 0, "L4x2": 0, "T4": 0, "T4x2": 0, "T4x4": 0}
        We suggest not including types with 0 machines (which might cause trouble).
        """
        # estimate the typical number of layers on node
        typical_layers_dict = {"A100": 12, "V100": 4, "L4": 8, "L4x2": 14, "T4": 4, "T4x2": 10, "T4x4": 20}
        total_layer_capacity = 0
        for machine_name in num_machines_dict:
            total_layer_capacity += num_machines_dict[machine_name] * typical_layers_dict[machine_name]
        if total_layer_capacity < LLaMa1_30B_TOTAL_LAYERS * 1.2:
            typical_layers_dict = {"A100": 18, "V100": 7, "L4": 11, "L4x2": 22, "T4": 7, "T4x2": 15, "T4x4": 30}
        typical_layers_dict = {m_type: typical_layers_dict[m_type] for m_type in num_machines_dict}

        # estimate the normalized performance
        normalized_perf_dict = {"A100": 48, "V100": 25, "L4": 9, "L4x2": 15, "T4": 8, "T4x2": 15, "T4x4": 23}
        normalized_perf_dict = {m_type: normalized_perf_dict[m_type] for m_type in num_machines_dict}
        for iteration in range(10):
            new_normalized_perf_dict = {}
            if "T4" in num_machines_dict:
                t4 = LLaMa30BonT4(num_machines_dict=num_machines_dict,
                                  typical_layers_dict=typical_layers_dict,
                                  normalized_perf_dict=normalized_perf_dict)
                t4_typical_tp = t4.get_typical_token_throughput(num_on_node_layers=typical_layers_dict["T4"])
                new_normalized_perf_dict["T4"] = t4_typical_tp * typical_layers_dict["T4"]
            if "T4x2" in num_machines_dict:
                t4x2 = LLaMa30BonT4x2(num_machines_dict=num_machines_dict,
                                      typical_layers_dict=typical_layers_dict,
                                      normalized_perf_dict=normalized_perf_dict)
                t4x2_typical_tp = t4x2.get_typical_token_throughput(num_on_node_layers=typical_layers_dict["T4x2"])
                new_normalized_perf_dict["T4x2"] = t4x2_typical_tp * typical_layers_dict["T4x2"]
            if "T4x4" in num_machines_dict:
                t4x4 = LLaMa30BonT4x4(num_machines_dict=num_machines_dict,
                                      typical_layers_dict=typical_layers_dict,
                                      normalized_perf_dict=normalized_perf_dict)
                t4x4_typical_tp = t4x4.get_typical_token_throughput(num_on_node_layers=typical_layers_dict["T4x4"])
                new_normalized_perf_dict["T4x4"] = t4x4_typical_tp * typical_layers_dict["T4x4"]
            if "L4" in num_machines_dict:
                l4 = LLaMa30BonL4(num_machines_dict=num_machines_dict,
                                  typical_layers_dict=typical_layers_dict,
                                  normalized_perf_dict=normalized_perf_dict)
                l4_typical_tp = l4.get_typical_token_throughput(num_on_node_layers=typical_layers_dict["L4"])
                new_normalized_perf_dict["L4"] = l4_typical_tp * typical_layers_dict["L4"]
            if "L4x2" in num_machines_dict:
                l4x2 = LLaMa30BonL4x2(num_machines_dict=num_machines_dict,
                                      typical_layers_dict=typical_layers_dict,
                                      normalized_perf_dict=normalized_perf_dict)
                l4x2_typical_tp = l4x2.get_typical_token_throughput(num_on_node_layers=typical_layers_dict["L4x2"])
                new_normalized_perf_dict["L4x2"] = l4x2_typical_tp * typical_layers_dict["L4x2"]
            if "V100" in num_machines_dict:
                v100 = LLaMa30BonV100(num_machines_dict=num_machines_dict,
                                      typical_layers_dict=typical_layers_dict,
                                      normalized_perf_dict=normalized_perf_dict)
                v100_typical_tp = v100.get_typical_token_throughput(num_on_node_layers=typical_layers_dict["V100"])
                new_normalized_perf_dict["V100"] = v100_typical_tp * typical_layers_dict["V100"]
            if "A100" in num_machines_dict:
                a100 = LLaMa30BonA100(num_machines_dict=num_machines_dict,
                                      typical_layers_dict=typical_layers_dict,
                                      normalized_perf_dict=normalized_perf_dict)
                a100_typical_tp = a100.get_typical_token_throughput(num_on_node_layers=typical_layers_dict["A100"])
                new_normalized_perf_dict["A100"] = a100_typical_tp * typical_layers_dict["A100"]
            normalized_perf_dict = new_normalized_perf_dict

        # save the final results
        self.num_machines_dict: Dict[str, int] = num_machines_dict
        self.typical_layers_dict: Dict[str, int] = typical_layers_dict
        self.normalized_perf_dict: Dict[str, float] = normalized_perf_dict

        # machine profiling results
        if "T4" in num_machines_dict:
            self.t4 = LLaMa30BonT4(num_machines_dict=num_machines_dict,
                                   typical_layers_dict=typical_layers_dict,
                                   normalized_perf_dict=normalized_perf_dict)
        else:
            self.t4 = None
        if "T4x2" in num_machines_dict:
            self.t4x2 = LLaMa30BonT4x2(num_machines_dict=num_machines_dict,
                                       typical_layers_dict=typical_layers_dict,
                                       normalized_perf_dict=normalized_perf_dict)
        else:
            self.t4x2 = None
        if "T4x4" in num_machines_dict:
            self.t4x4 = LLaMa30BonT4x4(num_machines_dict=num_machines_dict,
                                       typical_layers_dict=typical_layers_dict,
                                       normalized_perf_dict=normalized_perf_dict)
        else:
            self.t4x4 = None
        if "L4" in num_machines_dict:
            self.l4 = LLaMa30BonL4(num_machines_dict=num_machines_dict,
                                   typical_layers_dict=typical_layers_dict,
                                   normalized_perf_dict=normalized_perf_dict)
        else:
            self.l4 = None
        if "L4x2" in num_machines_dict:
            self.l4x2 = LLaMa30BonL4x2(num_machines_dict=num_machines_dict,
                                       typical_layers_dict=typical_layers_dict,
                                       normalized_perf_dict=normalized_perf_dict)
        else:
            self.l4x2 = None
        if "V100" in num_machines_dict:
            self.v100 = LLaMa30BonV100(num_machines_dict=num_machines_dict,
                                       typical_layers_dict=typical_layers_dict,
                                       normalized_perf_dict=normalized_perf_dict)
        else:
            self.v100 = None
        if "A100" in num_machines_dict:
            self.a100 = LLaMa30BonA100(num_machines_dict=num_machines_dict,
                                       typical_layers_dict=typical_layers_dict,
                                       normalized_perf_dict=normalized_perf_dict)
        else:
            self.a100 = None

        # model statistics
        self.token_size: float = 2 * Byte
        self.activation_size: float = 6656 * 2 * Byte
        self.model_param_sizes = [1 * GB] * 60
        assert len(self.model_param_sizes) == LLaMa1_30B_TOTAL_LAYERS, "Total layer number mismatch!"

    def check_type_exist(self, machine_type: str) -> bool:
        """
        Check if the given machine type exists in the current cluster.

        :param machine_type: machine type
        :return: True if the given machine type exists in the current model, False otherwise
        """
        return machine_type in self.num_machines_dict

    def get_profiling_results(self, machine_type: str) -> MachineProfile:
        """
        Get the profiling results of running one layer of the model on given type of machine.

        :param machine_type: machine type
        :return: MachineProfile
        """
        assert self.check_type_exist(machine_type), "Machine type not found!"
        if machine_type == "T4":
            return self.t4.get_profiling_results()
        elif machine_type == "T4x2":
            return self.t4x2.get_profiling_results()
        elif machine_type == "T4x4":
            return self.t4x4.get_profiling_results()
        elif machine_type == "L4":
            return self.l4.get_profiling_results()
        elif machine_type == "L4x2":
            return self.l4x2.get_profiling_results()
        elif machine_type == "V100":
            return self.v100.get_profiling_results()
        elif machine_type == "A100":
            return self.a100.get_profiling_results()
        else:
            assert False, "Found unknown machine type"

    def get_max_num_layers(self, machine_type: str) -> int:
        """
        Get the max number of layers the given type of machine can hold.

        :param machine_type: machine type
        :return: max number of layers the given type of machine can hold
        """
        assert self.check_type_exist(machine_type), "Machine type not found!"
        if machine_type == "T4":
            return self.t4.get_max_num_layers()
        elif machine_type == "T4x2":
            return self.t4x2.get_max_num_layers()
        elif machine_type == "T4x4":
            return self.t4x4.get_max_num_layers()
        elif machine_type == "L4":
            return self.l4.get_max_num_layers()
        elif machine_type == "L4x2":
            return self.l4x2.get_max_num_layers()
        elif machine_type == "V100":
            return self.v100.get_max_num_layers()
        elif machine_type == "A100":
            return self.a100.get_max_num_layers()
        else:
            assert False, "Found unknown machine type"

    def get_inference_settings(self, machine_type: str, num_on_node_layers: int) -> InferenceSettings:
        """
        Get the inference settings of the given machine type when there are given number of layers.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: InferenceSettings
        """
        assert self.check_type_exist(machine_type), "Machine type not found!"
        if machine_type == "T4":
            return self.t4.get_inference_settings(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x2":
            return self.t4x2.get_inference_settings(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x4":
            return self.t4x4.get_inference_settings(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4":
            return self.l4.get_inference_settings(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4x2":
            return self.l4x2.get_inference_settings(num_on_node_layers=num_on_node_layers)
        elif machine_type == "V100":
            return self.v100.get_inference_settings(num_on_node_layers=num_on_node_layers)
        elif machine_type == "A100":
            return self.a100.get_inference_settings(num_on_node_layers=num_on_node_layers)
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
        assert self.check_type_exist(machine_type), "Machine type not found!"
        if machine_type == "T4":
            return self.t4.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x2":
            return self.t4x2.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x4":
            return self.t4x4.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4":
            return self.l4.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4x2":
            return self.l4x2.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        elif machine_type == "V100":
            return self.v100.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        elif machine_type == "A100":
            return self.a100.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        else:
            assert False, "Found unknown machine type"

    def get_kv_cache_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the kv cache capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: kv cache capacity
        """
        assert self.check_type_exist(machine_type), "Machine type not found!"
        if machine_type == "T4":
            return self.t4.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x2":
            return self.t4x2.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x4":
            return self.t4x4.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4":
            return self.l4.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4x2":
            return self.l4x2.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "V100":
            return self.v100.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "A100":
            return self.a100.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        else:
            assert False, "Found unknown machine type"

    def get_activation_backup_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """
        Get the activation backup capacity of given machine type when using the current model.

        :param machine_type: machine type
        :param num_on_node_layers: number of layers on node
        :return: activation backup capacity
        """
        assert self.check_type_exist(machine_type), "Machine type not found!"
        if machine_type == "T4":
            return self.t4.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x2":
            return self.t4x2.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "T4x4":
            return self.t4x4.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4":
            return self.l4.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "L4x2":
            return self.l4x2.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "V100":
            return self.v100.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        elif machine_type == "A100":
            return self.a100.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
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
