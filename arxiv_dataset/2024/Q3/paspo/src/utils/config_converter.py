"""Module providing the ConfigConverter Class."""

import os
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any
from omegaconf import DictConfig, OmegaConf
import yaml
import torch
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog
from ray.air import CheckpointConfig
from financial_markets_gym.envs.financial_markets_env import FinancialMarketsEnv


class ConfigConverter:
    """This class converts a .yaml config into a usable ray config format"""

    logger = None

    @staticmethod
    def is_scientific_notation(input_string):
        """
        Checks if a string is passed in scientific notation, i.e. 5e-4 and returns a bool.
        :param input_string:
        :return:
        """
        if isinstance(input_string, str):
            pattern = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
            return bool(pattern.match(input_string))
        else:
            return False

    @staticmethod
    def generate_ray_config_omegaconf(cfg: DictConfig) -> Dict:
        dict_run_config = OmegaConf.to_container(cfg, resolve=True)
        
        dict_run_config = ConfigConverter.update_dict_format(
            dict_run_config=dict_run_config
        )
        
        dict_run_config = ConfigConverter.register_custom_env(
            dict_run_config=dict_run_config
        )

        dict_run_config = ConfigConverter.register_custom_action_distribution(
            dict_run_config=dict_run_config
        )

        dict_run_config = ConfigConverter.register_custom_algorithm(
            dict_run_config=dict_run_config
        )

        dict_run_config = ConfigConverter.register_custom_model(
            dict_run_config=dict_run_config
        )

        dict_run_config = ConfigConverter.update_config_custom_callbacks(
            dict_run_config=dict_run_config
        )

        dict_run_config = ConfigConverter.update_run_custom_callbacks(
            dict_run_config=dict_run_config,
            group_name=dict_run_config["group_name_wandb"],
            project_name=os.getenv("PROJECT_NAME"),
        )

        dict_run_config = ConfigConverter.update_experiment_name(
            dict_run_config=dict_run_config, run_name=dict_run_config["group_name_wandb"]
        )

        return dict_run_config

    @staticmethod
    def update_experiment_name(dict_run_config, run_name):
        """
        Sets an experiment name in the config
        :param dict_run_config:
        :param run_name:
        :return:
        """
        experiment_name = f"{run_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
        dict_run_config["name"] = experiment_name
        return dict_run_config

    @staticmethod
    def update_run_custom_callbacks(
        dict_run_config: Dict, group_name: str, project_name: str
    ) -> Dict:
        """# Update custom callbacks
        This updates the logger callbacks that handle the Weights&Biases Logging
        :param run_config_input:
        :param group_name:
        :param project_name:
        :return:
        """

        if "callbacks" in dict_run_config:
            tmp_callback_list = dict_run_config.get("callbacks")
            dict_run_config["callbacks"] = []
            for callback_entry in tmp_callback_list:
                if callback_entry == "WandbLoggerCallback":
                    from ray.tune.integration.wandb import WandbLoggerCallback

                    dict_run_config["callbacks"].append(
                        WandbLoggerCallback(
                            api_key=os.getenv("WANDB_API_KEY"),
                            project=project_name,
                            group=group_name,
                        )
                    )

                elif callback_entry == "WandbLoggerCustomCallback":
                    from utils.custom_wandb_logger import WandbLoggerCustomCallback

                    dict_run_config["callbacks"].append(
                        WandbLoggerCustomCallback(
                            api_key=os.getenv("WANDB_API_KEY"),
                            project=project_name,
                            group=group_name,
                        )
                    )
        return dict_run_config

    @staticmethod
    def update_config_custom_callbacks(dict_run_config: Dict) -> Dict:
        """
        This updates the logger callbacks that are defined within the config
        :param dict_run_config:
        :return:
        """
        logger = ConfigConverter.get_static_logger()
        if "callbacks" in dict_run_config.get("config"):
            callback_name = dict_run_config.get("config").get("callbacks")
            if callback_name == "EvaluationLoggerCallback":
                from utils.custom_callback import EvaluationLoggerCallback

                dict_run_config["config"]["callbacks"] = EvaluationLoggerCallback
                logger.info(f"Successfully added the callback '{callback_name}'")
            else:
                error = ValueError(f'Unknown callback "{callback_name}"')
                logger.error(error)
                raise error
        else:
            logger.warning(f"No custom callback present in run config")
        return dict_run_config


    @staticmethod
    def calculate_total_number_of_experiments(dict_run_config) -> int:
        """
        Checks for all the hyperparameter settings (associated with the 'grid_search' keyword)
        The grid size is the product of all entries
        :param dict_run_config:
        :return:
        """
        list_grid_search_sizes = ConfigConverter.check_dict_for_grid_search(
            dict_to_scan=dict_run_config
        )

        amount_experiments = 1
        for entry in list_grid_search_sizes:
            amount_experiments *= entry

        return amount_experiments

    @staticmethod
    def check_dict_for_grid_search(
        dict_to_scan: Dict, list_grid_search_sizes: List = None
    ) -> List:
        """
        Checks in a dict for the key 'grid_Search' and counts the length of the associated list value
        :param dict_to_scan:
        :param list_grid_search_sizes:
        :return:
        """
        if list_grid_search_sizes is None:
            list_grid_search_sizes = []

        for key, value in dict_to_scan.items():
            if key == "grid_search":
                # value is a list of values
                list_grid_search_sizes.append(len(value))
                break
            if isinstance(value, dict):
                ConfigConverter.check_dict_for_grid_search(
                    value, list_grid_search_sizes
                )
        return list_grid_search_sizes

    @staticmethod
    def update_scientific_notation_to_float(dict_to_scan: Dict) -> Dict:
        """
        Recursively goes through all entries in the dict and converts scientific notation into a decimal
        float if necessary
        :param dict_to_scan:
        :return:
        """
        for key, value in dict_to_scan.items():
            if isinstance(value, str):
                if ConfigConverter.is_scientific_notation(value):
                    dict_to_scan[key] = float(value)  # converts str into float
            elif isinstance(value, list):
                if any(
                    ConfigConverter.is_scientific_notation(list_entry)
                    for list_entry in value
                ):
                    dict_to_scan[key] = [float(list_entry) for list_entry in value]
            elif isinstance(value, dict):
                ConfigConverter.update_scientific_notation_to_float(value)
        return dict_to_scan


    @staticmethod
    def register_custom_model(dict_run_config):
        """
        registers a custom model with RLlibs Model Catalog
        :param dict_run_config:
        :return:
        """
        logger = ConfigConverter.get_static_logger()
        if "custom_model" in dict_run_config.get("config").get("model"):
            custom_model_name = (
                dict_run_config.get("config").get("model").get("custom_model")
            )

            if custom_model_name == "AutoRegressiveModel":
                from models.autoregressive_model import TorchAutoregressiveActionModel

                ModelCatalog.register_custom_model("AutoRegressiveModel", TorchAutoregressiveActionModel)

                logger.info(f'Successfully registered "{custom_model_name}"')
                
            elif custom_model_name == "CostValueFunctionCustomModel":
                from models.cost_vf_custom_model import (
                    CostValueFunctionCustomModel,
                )

                ModelCatalog.register_custom_model(
                    "CostValueFunctionCustomModel", CostValueFunctionCustomModel
                )

                logger.info(f'Successfully registered "{custom_model_name}"')
            elif custom_model_name == "LambdaCustomModel":
                from models.lambda_custom_model import LambdaCustomModel

                ModelCatalog.register_custom_model(
                    "LambdaCustomModel", LambdaCustomModel
                )
                logger.info(f'Successfully registered "{custom_model_name}"')
            elif custom_model_name == "CPOCustomModel":
                from models.cpo_model import CPOModel
                
                ModelCatalog.register_custom_model(
                    "CPOCustomModel", CPOModel
                )
                logger.info(f'Successfully registered "{custom_model_name}"')
            else:
                error = NotImplementedError(f'Unknown custom model "{custom_model_name}"')
                logger.error(error)
                raise error
        else:
            logger.warning(f"No custom model present in run config")
            
        #do check that ModelCatalog is patched
        import gym.spaces
        model_cls_fcnet = ModelCatalog._get_v2_model_class(gym.spaces.Box(0, 1, shape=(10,)), model_config={}, framework="torch")
        from models.custom_fc_model import CustomFullyConnectedNetwork
        if not model_cls_fcnet == CustomFullyConnectedNetwork:
            raise RuntimeError("ModelCatalog is not patched correctly. See README.md for instructions.")
            
        return dict_run_config

    @staticmethod
    def register_custom_algorithm(dict_run_config: Dict) -> Dict:
        """
        Registers a custom algorithm
        :param run_config_readable:
        :return:
        """

        logger = ConfigConverter.get_static_logger()

        algorithm_name = dict_run_config.get("run_or_experiment")

        if algorithm_name == "AutoregressivePPO":
            from algorithms.autoregressive_ppo_algorithm import (
                AutoregressivePPO,
            )
            dict_run_config["run_or_experiment"] = AutoregressivePPO
            return dict_run_config
        elif algorithm_name == "P3OPPO":
            from algorithms.P3O_ppo_algorithm import P3OPPO

            dict_run_config["run_or_experiment"] = P3OPPO
            return dict_run_config
        elif algorithm_name == "LagrangePPO":
            from algorithms.lagrange_ppo_algorithm import LagrangePPO

            dict_run_config["run_or_experiment"] = LagrangePPO
            return dict_run_config
        elif algorithm_name == "IPOPPO":
            from algorithms.ipo_ppo_algorithm import IPOPPO
            dict_run_config["run_or_experiment"] = IPOPPO
            return dict_run_config
          
        elif algorithm_name == "OptLayerPPO":
            from algorithms.optlayer_ppo_algorithm import OptLayerPPOAlgorithm

            dict_run_config["run_or_experiment"] = OptLayerPPOAlgorithm
            return dict_run_config
        elif algorithm_name == "CUP":
            from algorithms.cup_ppo_algorithm import CUPAlgorithm
            dict_run_config["run_or_experiment"] = CUPAlgorithm
            return dict_run_config
        elif algorithm_name == "CPO":
            from algorithms.cpo_algorithm import CPOAlgorithm
            
            dict_run_config["run_or_experiment"] = CPOAlgorithm
            return dict_run_config
        
        return dict_run_config

    @staticmethod
    def register_custom_action_distribution(dict_run_config: Dict) -> Dict:
        """
        Registers a custom action distribution to RLlib's Model Catalog
        :param run_config_input:
        :return:
        """

        logger = ConfigConverter.get_static_logger()

        if "custom_action_dist" in dict_run_config.get("config").get("model"):
            custom_dist_name = (
                dict_run_config.get("config").get("model").get("custom_action_dist")
            )

            if custom_dist_name == "TorchDirichletCustom":
                from action_distributions.distribution_dirichlet_custom import (
                    TorchDirichletCustom,
                )

                ModelCatalog.register_custom_action_dist(
                    "TorchDirichletCustom", TorchDirichletCustom
                )
            elif custom_dist_name == "TorchDirichletCustomStable":
                from action_distributions.distribution_dirichlet_custom import (
                    TorchDirichletCustomStable,
                )

                ModelCatalog.register_custom_action_dist(
                    "TorchDirichletCustomStable", TorchDirichletCustomStable
                )
            elif custom_dist_name == "AutoregressiveDistribution":
                from action_distributions.generic_autoregressive_distribution import TorchAutoregressiveGenericDistribution
                
                ModelCatalog.register_custom_action_dist(
                    "AutoregressiveDistribution", TorchAutoregressiveGenericDistribution
                )
            else:
                error = NotImplementedError(
                    f'Distribution "{custom_dist_name}" not implemented'
                )
                logger.error(
                    f"Custom Distribution '{custom_dist_name}' has no implementation to be registered."
                )
                raise error

        return dict_run_config


    @staticmethod
    def register_custom_env(dict_run_config: Dict) -> Dict:
        """
        This method registers a custom env for RLlib
        :param dict_run_config:
        :return:
        """
        logger = ConfigConverter.get_static_logger()

        select_env = dict_run_config.get("config").get("env")

        
        if select_env == "financial-markets-env-short-selling-v0":
            dict_run_config["config"]["env_config"]["seed"] = dict_run_config["config"]["seed"]
            if dict_run_config["config"]["env_config"]["use_frame_stack"]:
                from gym.wrappers import FrameStack, FlattenObservation
                num_stack = dict_run_config["config"]["env_config"]["num_stack"]
                register_env(
                    select_env,
                    lambda config: FlattenObservation(FrameStack(FinancialMarketsEnv(**config), num_stack=num_stack))
                )
            else:
                register_env(
                    select_env,
                    lambda config: FinancialMarketsEnv(**config)
                )
        elif select_env == "compute":
            from iot_computation_gym.envs.iot_computation_env import IOTComputationEnv
            
            register_env(select_env, lambda config: IOTComputationEnv(**config))
        elif select_env == "synth":
            from envs.synth_env import SynthEnv            
            register_env(select_env, lambda config: SynthEnv(n_dimension=config["n_dimensions"], n_states=config["n_states"], env_gen_seed=config["env_gen_seed"], use_mlp_reward_function=config["use_mlp_reward_function"]))            
        return dict_run_config


    @staticmethod
    def update_dict_format(dict_run_config: Dict) -> Dict:
        """
        This modifies the original (legacy) format to guarantee backward compatibility with the older ray format
        :param dict_run_config: dict of the run config
        :return:
        """

        # Log the original input for reference
        logger = ConfigConverter.get_static_logger()

        if "config" not in dict_run_config:
            dict_run_config["config"] = {}
            logger.info("Added 'config' key to run_config_input.")

        if "env" in dict_run_config:
            dict_run_config["config"]["env"] = dict_run_config.pop("env")
            logger.info("Moved 'env' key to 'config'.")

        if "run" in dict_run_config:
            dict_run_config["run_or_experiment"] = dict_run_config.pop("run")
            logger.info("Renamed 'run' to 'run_or_experiment'.")

        dict_run_config = ConfigConverter.update_scientific_notation_to_float(
            dict_to_scan=dict_run_config
        )

        return dict_run_config

    @staticmethod
    def get_static_logger():
        if ConfigConverter.logger is None:
            ConfigConverter.logger = logging.getLogger(os.getenv("LOGGER_NAME"))
        return ConfigConverter.logger

