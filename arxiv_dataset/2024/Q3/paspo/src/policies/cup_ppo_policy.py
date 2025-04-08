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
from policies.P3O_ppo_policy import ValueInclCostNetworkMixin

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class CUPTorchPolicy(
    # ValueNetworkMixin, -> we overwrite the extra_action_out manually here
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

        A, b = load_polytope(n_dim=n_dimensions,
                             storage_method=config["model"]["custom_model_config"]["polytope_storage_method"],
                             generation_method=config["model"]["custom_model_config"]["polytope_generation_method"],
                             polytope_generation_data=config["model"]["custom_model_config"][
                                 "polytope_generation_data"])

        self.A = A
        self.b = b

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        # ValueNetworkMixin.__init__(self, config)
        ValueInclCostNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        self.gamma = config["gamma"]
        self.lambda_ = config["lambda"]
        
        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    def loss_pi_cost(self,
                     model: ModelV2,
                     dist_class: Type[ActionDistribution],
                     train_batch: SampleBatch):

        number_constraints = self.A.shape[0]
        torch_gae_matrix = torch.stack([train_batch[f"cost_advantages_{i}"] for i in range(number_constraints)], dim=1)
        lambda_weighted_gae = model.lambda_penalty_model.forward(torch_gae_matrix)

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        if self.gamma < 1:
            coef_proj = (1 - self.gamma * self.lambda_) / (
                1 - self.gamma
            )
        else:
            coef_proj = 1

        # original pre training distriution
        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )
        action_kl = prev_action_dist.kl(curr_action_dist)

        # calculate the loss -> we perform a minimization
        surrogate_loss = torch.mean(
                lambda_weighted_gae * logp_ratio * coef_proj + action_kl
        )

        return surrogate_loss

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

        """
        # 1) Policy Improvement ###
        print(type(train_batch))
        print(train_batch)
        print("$$$$$$$$$$$4")
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # classical gradient clipping for CUP, the advantages are GAE:
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        policy_improvement_loss = reduce_mean_valid(-surrogate_loss)

        #do a gradient step for the policy, in RllIb the policy stores the optimizer for the model network
        self.model.policy_improvement_optimizer.zero_grad()
        policy_improvement_loss.backward(retain_graph=True)
        self.model.policy_improvement_optimizer.step()
        
        #generate new computational graph for the train_batch inputs
        for key, value in train_batch.items():
            if torch.is_tensor(value):
                detached_tensor = value.detach()
                # Clear gradients for the detached tensor
                detached_tensor.grad = None
                train_batch[key] = detached_tensor
        """ or None
        # 2) Projection
        # use model with updated parameters
        #cost advantages in batch are GAE

        # 2.1) Train Lagrange
        number_constraints = self.A.shape[0]

        torch_penalty_matrix = torch.stack([train_batch[f"cost_obs_{i}"] for i in range(number_constraints)], dim=1)
        #training of the lagrange multiplier
        model.lambda_penalty_model.update_lagrange_multiplier(torch_penalty_matrix)

        # 2.2) Do Projection
        """
        torch_gae_matrix = torch.stack([train_batch[f"cost_advantages_{i}"] for i in range(number_constraints)], dim=1)
        lambda_weighted_gae = model.lambda_penalty_model.forward(torch_gae_matrix)

        logits, state = model(train_batch)
        curr_action_dist_proj = dist_class(logits, model)

        logp_ratio_proj = torch.exp(
            curr_action_dist_proj.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        coef_proj = 1

        #original pre training distriution
        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )
        action_kl_proj = prev_action_dist.kl(curr_action_dist_proj)

        #calculate the loss
        surrogate_loss_proj = (
            lambda_weighted_gae*logp_ratio_proj*coef_proj + action_kl_proj
        )

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # we are not tracking the kl divergence at this point in time # TODO
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        """ or None

        # 3-4) Update Value function V(s) and Cost Value function V_c(s)
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

        """
        # Adding cost surrogate
        #kappa = self.config["model"]["custom_model_config"]["init_kappa"]
        cost_surrogate_loss_total = 0
        for idx_cost in range(model.amount_constraints):
            cost_surrogate_loss = torch.max(
                train_batch[f'{Postprocessing_Custom.COST_ADVANTAGES}{idx_cost}'] * logp_ratio,
                train_batch[f'{Postprocessing_Custom.COST_ADVANTAGES}{idx_cost}']
                * torch.clamp(
                    logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
                ),
            ) + (1 - self.config["gamma"]) * (train_batch[
                                                  f'{Postprocessing_Custom.COST_VALUE_TARGETS}{idx_cost}'] - 0)  # (....)
            cost_surrogate_loss_total = cost_surrogate_loss_total + \
                                        torch.maximum(torch.zeros_like(cost_surrogate_loss),
                                                              cost_surrogate_loss)

        # we need to incoporate the cost_surrogate_loss for multiple entries /However cost_surrogate_total_is our run variable
        surrogate_loss = surrogate_loss - cost_surrogate_loss_total
        """ or None
        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
            if self.config["model"]["custom_model_config"][
                "cost_vf_use_critic"]:  # Training of the value functions of cost vf
                for idx_cost in range(model.amount_constraints):
                    cost_value_fn_out = model.cost_value_function(idx_cost)
                    cost_vf_loss = torch.pow(
                        cost_value_fn_out - train_batch[f'{Postprocessing_Custom.COST_VALUE_TARGETS}{idx_cost}'], 2.0
                    )
                    cost_vf_loss_clipped = torch.clamp(cost_vf_loss, 0, self.config["vf_clip_param"])
                    vf_loss_clipped = vf_loss_clipped + cost_vf_loss_clipped  # adding the cost_vf_loss_clipped

        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0

        #the total_loss is minimized so we need to minimize the NEGATIVE surrogate loss (to maximize the Advantage)
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


    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:
        """
        Overwriting the learn on batch process
        :param postprocessed_batch:
        :return:
        """
        #
        #print(self.model)
        #print(self.dist_class)

        fetches = super(TorchPolicyV2, self).learn_on_batch(samples=postprocessed_batch)

        #training of the
        loss_pi_cost = self.loss_pi_cost(model=self.model, dist_class=self.dist_class, train_batch=postprocessed_batch)

        self._optimizers[0].zero_grad()
        loss_pi_cost.backward(retain_graph=True)
        self._optimizers[0].step()

        return fetches
        """
        # Set Model to train mode.
        if self.model:
            self.model.train()
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

        return fetches
        """

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
