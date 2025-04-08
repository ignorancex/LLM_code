#define enum with the different modes of the optlayer
from enum import Enum
class OptLayerMode(Enum):
    CP = 1 #learn on actions that can violate
    CC = 2 #learn on actions that cannot violate
    CPC = 3 #mixture of the two

import torch
import functools
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy





#when we compute the action we need to raw action and the action from optlayer as well as the cost

def _compute_action_helper(
    policy, input_dict, state_batches, seq_lens, explore, timestep
):
    """Shared forward pass logic (w/ and w/o trajectory view API).

    Returns:
        A tuple consisting of a) actions, b) state_out, c) extra_fetches.
    """
    explore = explore if explore is not None else policy.config["explore"]
    timestep = timestep if timestep is not None else policy.global_timestep
    policy._is_recurrent = state_batches is not None and state_batches != []

    # Switch to eval mode.
    if policy.model:
        policy.model.eval()

    if is_overridden(policy.action_sampler_fn) and policy.action_sampler_fn is not None:
        action_dist = dist_inputs = None
        actions, logp, state_out = policy.action_sampler_fn(
            policy.model,
            obs_batch=input_dict,
            state_batches=state_batches,
            explore=explore,
            timestep=timestep,
        )
    else:
        # Call the exploration before_compute_actions hook.
        policy.exploration.before_compute_actions(explore=explore, timestep=timestep)
        if is_overridden(policy.action_distribution_fn):
            try:
                dist_inputs, dist_class, state_out = policy.action_distribution_fn(
                    policy.model,
                    obs_batch=input_dict,
                    state_batches=state_batches,
                    seq_lens=seq_lens,
                    explore=explore,
                    timestep=timestep,
                    is_training=False,
                )
            except TypeError as e:
                if (
                    "positional argument" in e.args[0]
                    or "unexpected keyword argument" in e.args[0]
                ):
                    dist_class = policy.dist_class
                    dist_inputs, state_out = policy.model(input_dict, state_batches, seq_lens)
                else:
                    raise e

        else:
            dist_class = policy.dist_class
            dist_inputs, state_out = policy.model(input_dict, state_batches, seq_lens)

        if not (
            isinstance(dist_class, functools.partial)
            or issubclass(dist_class, TorchDistributionWrapper)
        ):
            raise ValueError(
                "`dist_class` ({}) not a TorchDistributionWrapper "
                "subclass! Make sure your `action_distribution_fn` or "
                "`make_model_and_action_dist` return a correct "
                "distribution class.".format(dist_class.__name__)
            )
        action_dist = dist_class(dist_inputs, policy.model)

        # Get the exploration action from the forward results.
        actions, logp = policy.exploration.get_exploration_action(
            action_distribution=action_dist, timestep=timestep, explore=explore
        )

    if policy.optlayer_mode == OptLayerMode.CPC:
        cost = policy.opt_layer.compute_cost(actions)
    safe_actions = policy.opt_layer(actions)
    unsafe_actions = actions
    
    if policy.constrained:
        actions = safe_actions

    
    input_dict[SampleBatch.ACTIONS] = actions

    # Add default and custom fetches.
    extra_fetches = policy.extra_action_out(
        input_dict, state_batches, policy.model, action_dist
    )
    
    if policy.optlayer_mode == OptLayerMode.CPC:
        # Add the cost to the extra fetches
        extra_fetches["COST"] = cost
    
    # Add the safe actions to the extra fetches
    extra_fetches["SAFE_ACTIONS"] = safe_actions
    
    # Add the unsafe actions to the extra fetches
    extra_fetches["UNSAFE_ACTIONS"] = unsafe_actions

    # Action-dist inputs.
    if dist_inputs is not None:
        extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

    # Action-logp and action-prob.
    if logp is not None:
        extra_fetches["ACTION_PROB_UNSAFE"] = torch.exp(logp.float())
        extra_fetches["ACTION_LOGP_UNSAFE"] = logp

        log_p_safe = action_dist.logp(safe_actions)
        extra_fetches["ACTION_PROB_SAFE"] = torch.exp(log_p_safe.float())
        extra_fetches["ACTION_LOGP_SAFE"] = log_p_safe
        
        #extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
        #extra_fetches[SampleBatch.ACTION_LOGP] = logp

    # Update our global timestep by the batch size.
    policy.global_timestep += len(input_dict[SampleBatch.CUR_OBS])

    return convert_to_numpy((actions, state_out, extra_fetches))
