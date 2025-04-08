"""Entry point of the python script."""

import os
import logging
from typing import List

import hydra
from ray.air import CheckpointConfig

from omegaconf import DictConfig
import ray
import sys

from ray.tune.experiment import Trial

from ray import air, tune
from dotenv import load_dotenv
from utils.config_converter import ConfigConverter

from ray.tune.callback import Callback

load_dotenv()  # take environment variables from .env.

@hydra.main(version_base=None, config_path="../run_config", config_name="config")
def main(cfg: DictConfig):
    if not ray.is_initialized():
        ray.init(local_mode=cfg["local_mode"], log_to_driver=False)

    ray_cfg = ConfigConverter.generate_ray_config_omegaconf(cfg)
    
    ray_cfg.pop("local_mode") #remove local_mode from ray config as it will crash otherwise
    
    trainable = ray_cfg.pop("run_or_experiment")
    dict_param_space = ray_cfg.pop("config")
    ray_cfg.pop("group_name_wandb")

    if "checkpoint_config" in ray_cfg:
        ray_cfg["checkpoint_config"] = CheckpointConfig(
            **ray_cfg.pop("checkpoint_config")
        )

    number_of_torch_threads = ray_cfg.pop("number_of_torch_threads")
    run_cfg = air.RunConfig(**ray_cfg)
        
    class SetupCallback(Callback):
            
        def on_trial_start(self, iteration: int, trials: List[Trial], trial: Trial, **info):
            import os
            affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(os.getpid(), affinity) 
            #this ensures that the process of the worker can run on all cores and is not bound by ray
                        
            import torch
            torch.set_num_threads(number_of_torch_threads)
            import numpy as np
            #this is an ugly hack to get the result of the first evaluation in wandb...
            from ray.air.callbacks.wandb import _QueueItem
            
            result = ray.get(trial.runner.evaluate.remote())
            constraint_violation_distance = result["evaluation"]["custom_metrics"]["constraint_violation_distance"]
        
            result["evaluation"]["custom_metrics"]["constraint_violation_distance_total"] = np.sum(constraint_violation_distance)
            result["evaluation"]["custom_metrics"]["constraint_violation_distance_mean"] = np.mean(constraint_violation_distance)
            
            result["evaluation"]["custom_metrics"]["bool_violation_total"] = np.sum(result["evaluation"]["custom_metrics"]["bool_violation"])
            result["evaluation"]["custom_metrics"]["bool_violation_mean"] = np.mean(result["evaluation"]["custom_metrics"]["bool_violation"])
            
            result["evaluation"]["custom_metrics"]["bool_violations_total"] = np.sum(result["evaluation"]["custom_metrics"]["bool_violations"])
            result["evaluation"]["custom_metrics"]["bool_violations_mean"] = np.mean(result["evaluation"]["custom_metrics"]["bool_violations"])

            result["timesteps_total"] = 0
            run_cfg.callbacks[0]._trial_queues[trial].put((_QueueItem.RESULT, result))

            
        def on_trial_restore(self, iteration: int, trials: List[Trial], trial: Trial, **info):
            import os
            affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(os.getpid(), affinity) 
            #this ensures that the process of the worker can run on all cores and is not bound by ray
            
            import torch
            torch.set_num_threads(number_of_torch_threads)


    run_cfg.callbacks.append(SetupCallback())
    
    # define tuner
    tuner = tune.Tuner(
        trainable=trainable, param_space=dict_param_space, run_config=run_cfg
    )

   
    # start policy training
    results = tuner.fit()



if __name__ == "__main__":
    main()
