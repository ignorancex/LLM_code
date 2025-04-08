import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)

from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton

from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import TensorType
from utils.custom_keys import SampleBatch_Custom, Postprocessing_Custom
from utils.polytope_loader import load_polytope


from utils.cost_postprocessing import compute_cost_values_P3O, compute_cost_gae_for_sample_batch

torch, nn = try_import_torch()

import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ValueInclCostNetworkMixin:
    """Assigns the `_value()` method to a TorchPolicy.
    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    -> Also added _value_cost
    """

    def __init__(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0].item()
        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0
        if config.get("model").get("custom_model_config").get("cost_vf_use_gae"):
            def cost_value(idx_cost, **input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.cost_value_function(idx_cost)[0].item()

        else:
            def cost_value(idx_cost, *args, **input_dict):
                return 0.0

        self._value = value #making the function available
        self._cost_value = cost_value #making the function available

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        """Defines extra fetches per action computation.
        Args:
            input_dict (Dict[str, TensorType]): The input dict used for the action
                computing forward pass.
            state_batches (List[TensorType]): List of state tensors (empty for
                non-RNNs).
            model (ModelV2): The Model object of the Policy.
            action_dist: The instantiated distribution
                object, resulting from the model's outputs and the given
                distribution class.
        Returns:
            Dict[str, TensorType]: Dict with extra tf fetches to perform per
                action computation.
        """
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        tmp_dict = {}
        tmp_dict[SampleBatch.VF_PREDS] = model.value_function()
        for i in range(model.amount_constraints):
            tmp_dict[f'{SampleBatch_Custom.COST_VF_PREDS}{i}'] = model.cost_value_function(i)
        return tmp_dict


class P3OPPOTorchPolicy(
    #ValueNetworkMixin, -> we overwrite the extra_action_out manually here
    ValueInclCostNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        validate_config(config)
        n_dimensions = action_space.shape[0]
        

        A,b = load_polytope(n_dim=n_dimensions,
                            storage_method=config["model"]["custom_model_config"]["polytope_storage_method"], 
                            generation_method=config["model"]["custom_model_config"]["polytope_generation_method"], 
                            polytope_generation_data=config["model"]["custom_model_config"]["polytope_generation_data"])
        
        self.A = A
        self.b = b


        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        #ValueNetworkMixin.__init__(self, config)
        ValueInclCostNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
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
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
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
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        #Adding cost surrogate
        kappa = self.config["model"]["custom_model_config"]["init_kappa"]
        cost_surrogate_loss_total = 0
        for idx_cost in range(model.amount_constraints):
            cost_surrogate_loss = torch.max(
                train_batch[f'{Postprocessing_Custom.COST_ADVANTAGES}{idx_cost}'] * logp_ratio,
                train_batch[f'{Postprocessing_Custom.COST_ADVANTAGES}{idx_cost}']
                * torch.clamp(
                    logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
                ),
            )
            
            cost_surrogate_loss = cost_surrogate_loss + (1-self.config["gamma"]) * train_batch["J_cost_"+str(idx_cost)] #TODO do not harcode cost limit of 0
            
            cost_surrogate_loss= cost_surrogate_loss.mean()
            
            
            cost_surrogate_loss_total = cost_surrogate_loss_total + \
                                        kappa * F.relu(cost_surrogate_loss)

        # we need to incoporate the cost_surrogate_loss for multiple entries /However cost_surrogate_total_is our run variable
        mean_policy_loss = mean_policy_loss + cost_surrogate_loss_total

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
            if self.config["model"]["custom_model_config"]["cost_vf_use_critic"]: #Training of the value functions of cost vf
                for idx_cost in range(model.amount_constraints):
                    cost_value_fn_out = model.cost_value_function(idx_cost)
                    cost_vf_loss = torch.pow(
                        cost_value_fn_out - train_batch[f'{Postprocessing_Custom.COST_VALUE_TARGETS}{idx_cost}'], 2.0
                    )
                    cost_vf_loss_clipped = torch.clamp(cost_vf_loss, 0, self.config["vf_clip_param"])
                    vf_loss_clipped = vf_loss_clipped + cost_vf_loss_clipped #adding the cost_vf_loss_clipped

        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0

        total_loss = mean_policy_loss + reduce_mean_valid(
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

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():

            sample_batch = compute_cost_values_P3O(
                self, sample_batch, other_agent_batches, episode)

            sample_batch = compute_cost_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)

            sample_batch = compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

            return sample_batch

    @override(TorchPolicyV2)
    def make_model(self):
        from models.cost_vf_custom_model import make_cost_model
        return make_cost_model(policy=self,
                               obs_space=self.observation_space,
                               action_space=self.action_space,
                               config=self.config)
