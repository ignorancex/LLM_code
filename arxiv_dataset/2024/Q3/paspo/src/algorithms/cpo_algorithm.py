from typing import Type
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict
from ray.rllib.policy.policy import Policy
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

import numpy as np


class CPOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CPOAlgorithm)

        
class CPOAlgorithm(PPO):
    _allow_unknown_configs = True
    
    #@override(Algorithm)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from policies.cpo_policy import CPOTorchPolicy

            return CPOTorchPolicy
        elif config["framework"] == "tf":
            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy

            return PPOTF1Policy
        else:
            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF2Policy

            return PPOTF2Policy

    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if self._by_agent_steps:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config["train_batch_size"]
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
            )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages and costs
        if self.config["standardize_advantages"]:
            train_batch = standardize_fields(train_batch, ["advantages"])
        
        if self.config["standardize_costs"]:
            train_batch = standardize_fields(train_batch, ["ADVANTAGES_COSTS"])
        
        # set episodes correctly for sample batch
        _, idx = np.unique(train_batch.policy_batches["default_policy"]["eps_id"], return_index=True)
        train_batch.policy_batches["default_policy"]["EPSIODE_COSTS"] = train_batch.policy_batches["default_policy"]["EPSIODE_COSTS"][idx].mean()[None].repeat(train_batch.policy_batches["default_policy"]["EPSIODE_COSTS"].shape)
        # Train
        if self.config["simple_optimizer"]:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # # For each policy: update KL scale and warn about possible issues
        # for policy_id, policy_info in train_results.items():
        #     # Update KL loss with dynamic scaling
        #     # for each (possibly multiagent) policy we are training
        #     kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
        #     self.get_policy(policy_id).update_kl(kl_divergence)

        #     # Warn about excessively high value function loss
        #     scaled_vf_loss = (
        #         self.config["vf_loss_coeff"] * policy_info[LEARNER_STATS_KEY]["vf_loss"]
        #     )
        #     policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
        #     if (
        #         log_once("ppo_warned_lr_ratio")
        #         and self.config.get("model", {}).get("vf_share_layers")
        #         and scaled_vf_loss > 100
        #     ):
        #         logger.warning(
        #             "The magnitude of your value function loss for policy: {} is "
        #             "extremely large ({}) compared to the policy loss ({}). This "
        #             "can prevent the policy from learning. Consider scaling down "
        #             "the VF loss by reducing vf_loss_coeff, or disabling "
        #             "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
        #         )
        #     # Warn about bad clipping configs.
        #     train_batch.policy_batches[policy_id].set_get_interceptor(None)
        #     mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
        #     if (
        #         log_once("ppo_warned_vf_clip")
        #         and mean_reward > self.config["vf_clip_param"]
        #     ):
        #         self.warned_vf_clip = True
        #         logger.warning(
        #             f"The mean reward returned from the environment is {mean_reward}"
        #             f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
        #             f" Consider increasing it for policy: {policy_id} to improve"
        #             " value function convergence."
        #         )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results
