from typing import Any, Dict, List, Optional, Tuple, Type, Union
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation import Episode
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_TRAINED

from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
import functools
from ray.rllib.policy.sample_batch import SampleBatch
import torch

from policies.optlayer_helpers import OptLayerMode, _compute_action_helper
from utils.opt_layer import OptLayer
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)

from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import AgentID

from ray.rllib.evaluation.postprocessing import discount_cumsum
from utils.polytope_loader import load_polytope

import numpy as np

class OptlayerPPOPolicy(PPOTorchPolicy):
    
    def __init__(self, observation_space, action_space, config):
        #whether to use the constrained action or not
        self.constrained = True #whether to sample constrained actions or not
        
        n_dimensions = action_space.shape[0]
        

        A,b = load_polytope(n_dim=n_dimensions,
                            storage_method=config["model"]["custom_model_config"]["polytope_storage_method"], 
                            generation_method=config["model"]["custom_model_config"]["polytope_generation_method"], 
                            polytope_generation_data=config["model"]["custom_model_config"]["polytope_generation_data"])
        
        self.A = A
        self.b = b
        
        self.opt_layer = OptLayer(n_dimensions, A, b, "cpu", normalize=config["model"]["custom_model_config"]["normalize_constraints"]) #TODO do not hardcode cpu

        if config["model"]["custom_model_config"]["optlayer_mode"] == "CP":
            self.optlayer_mode = OptLayerMode.CP
        elif config["model"]["custom_model_config"]["optlayer_mode"] == "CC":
            self.optlayer_mode = OptLayerMode.CC
        elif config["model"]["custom_model_config"]["optlayer_mode"] == "CPC":
            self.optlayer_mode = OptLayerMode.CPC
        else:
            raise ValueError("Invalid optlayer mode")
        
        self._update_using_safe_action = True
        self._update_using_costs = None #none for initialization
        
        super().__init__(observation_space, action_space, config)
        
    
        
    
    #when we compute the action we need to raw action and the action from optlayer as well as the cost
    
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        return _compute_action_helper(
            self, input_dict, state_batches, seq_lens, explore, timestep
        )
    
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        update_using_safe_action = self._update_using_safe_action
        update_using_costs = self._update_using_costs
        
        if update_using_costs is None: #init because of dummy loss and view requirements
            _ = train_batch["SAFE_ACTIONS"]
            _ = train_batch["UNSAFE_ACTIONS"]
            _ = train_batch["ACTION_LOGP_SAFE"]
            _ = train_batch["ACTION_LOGP_UNSAFE"]
            _ = train_batch["ADVANTAGES_COSTS"]
            _ = train_batch["VALUE_TARGETS_COSTS"]
            update_using_costs = False
        
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch["SAFE_ACTIONS" if update_using_safe_action else "UNSAFE_ACTIONS"])
            - train_batch["ACTION_LOGP_SAFE" if update_using_safe_action else "ACTION_LOGP_UNSAFE"]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch["ADVANTAGES_COSTS" if update_using_costs else Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch["ADVANTAGES_COSTS" if update_using_costs else Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch["VALUE_TARGETS_COSTS" if update_using_costs else Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            
            
            if self.optlayer_mode == OptLayerMode.CPC:
                sample_batch =  compute_gae_for_sample_batch_with_cost(
                    self, sample_batch, other_agent_batches, episode
                )
                
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:
        # Set Model to train mode.
        if self.model:
            self.model.train()
        
        self._update_using_costs = False

        if self.optlayer_mode == OptLayerMode.CPC:
            self._update_using_safe_action = False
            self._update_using_costs = True
        elif self.optlayer_mode == OptLayerMode.CP:
            self._update_using_safe_action = False
        elif self.optlayer_mode == OptLayerMode.CC:
            self._update_using_safe_action = True
        else:
            raise ValueError("Invalid optlayer mode")
        
        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=postprocessed_batch, result=learn_stats
        )

        # Compute gradients (will calculate all losses and `backward()`
        # them to get the grads).
        grads, fetches = self.compute_gradients(postprocessed_batch)

        # Step the optimizers.
        self.apply_gradients(_directStepOptimizerSingleton)

        if self.model:
            fetches["model"] = self.model.metrics()
        fetches.update(
            {
                "custom_metrics": learn_stats,
                NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count,
            }
        )
        
        #only cpc mode does a second step with the safe action
        if not self.optlayer_mode == OptLayerMode.CPC:
            return fetches
            
        self._update_using_costs = False
        self._update_using_safe_action = True
        
        # Callback handling.
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=postprocessed_batch, result=learn_stats
        )

        # Compute gradients (will calculate all losses and `backward()`
        # them to get the grads).
        grads, fetches_ = self.compute_gradients(postprocessed_batch)

        # Step the optimizers.
        self.apply_gradients(_directStepOptimizerSingleton)

        if self.model:
            fetches_["model"] = self.model.metrics()
        
        
        
        #fetches.update({"cpc_" + key : value for key, value in fetches_})
        fetches.update(fetches_) 
        fetches.update(
            {
                "custom_metrics": learn_stats,
                NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count,
            }
        )


        return fetches





def compute_advantages_with_costs(
    rollout: SampleBatch,
    last_r: float,
    gamma: float = 0.9,
    lambda_: float = 1.0,
    use_gae: bool = True,
    use_critic: bool = True,
):
    """Given a rollout, compute its value targets and the advantages.

    Args:
        rollout: SampleBatch of a single trajectory.
        last_r: Value estimation for last observation.
        gamma: Discount factor.
        lambda_: Parameter for GAE.
        use_gae: Using Generalized Advantage Estimation.
        use_critic: Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch with experience from rollout and processed rewards.
    """

    assert (
        SampleBatch.VF_PREDS in rollout or not use_critic
    ), "use_critic=True but values not found"
    assert use_critic or not use_gae, "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS], np.array([last_r])])
        delta_t = rollout[SampleBatch.REWARDS] - rollout["COST"] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout["ADVANTAGES_COSTS"] = discount_cumsum(delta_t, gamma * lambda_)
        rollout["VALUE_TARGETS_COSTS"] = (
            rollout["ADVANTAGES_COSTS"] + rollout[SampleBatch.VF_PREDS]
        ).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS] - rollout["COST"], np.array([last_r])]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if use_critic:
            rollout["ADVANTAGES_COSTS"] = (
                discounted_returns - rollout[SampleBatch.VF_PREDS]
            )
            rollout["VALUE_TARGETS_COSTS"] = discounted_returns
        else:
            rollout["ADVANTAGES_COSTS"] = discounted_returns
            rollout["VALUE_TARGETS_COSTS"] = np.zeros_like(
                rollout["ADVANTAGES_COSTS"]
            )

    rollout["ADVANTAGES_COSTS"] = rollout["ADVANTAGES_COSTS"].astype(
        np.float32
    )

    return rollout


def compute_gae_for_sample_batch_with_cost(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches: Optional dict of AgentIDs mapping to other
            agents' trajectory data (from the same episode).
            NOTE: The other agents use the same policy.
        episode: Optional multi-agent episode object in which the agents
            operated.

    Returns:
        The postprocessed, modified SampleBatch (or a new one).
    """

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        last_r = policy._value(**input_dict)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages_with_costs(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
    )

    return batch
