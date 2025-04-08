import torch
import logging
import os

from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
from typing import Dict, Optional

from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID

from ray.rllib.evaluation.postprocessing import discount_cumsum
from utils.custom_keys import Postprocessing_Custom, SampleBatch_Custom


logger = logging.getLogger(os.getenv("LOGGER_NAME"))




def calculate_constraint_violations(A, b, action):
    """
    Calculate the constraint violations
    :param A: the matrix A (n_constraints x n_dim)
    :param b: the vector b (n_donstraints)
    :param action: the action as a batch (n_batch x n_dim)
    :return: the constraint violations as a batch (n_batch x n_constraints)
    """
    
    return np.maximum(A @ action.T - b.reshape(-1, 1), 0).T
    

def check_constraint_violations(np_penalty_matrix: np.ndarray, sample_batch=None, action_processed=None):
    #return the number of constraint violations
    number_of_contraint_violations = (np_penalty_matrix > 1e-3).sum(axis=1) #total number of violated constraints #TODO do not hardcode tolerance
    return number_of_contraint_violations


## P3O ##
def calculate_penalty_score_P3O(policy, sample_batch):
    np_penalty_matrix = calculate_constraint_violations(policy.A, policy.b, sample_batch[SampleBatch.ACTIONS])

    number_constraint_violations = check_constraint_violations(
        np_penalty_matrix, sample_batch=sample_batch, action_processed=sample_batch[SampleBatch.ACTIONS]
    )

    unweighted_penalty_violation_score = np.sum(np_penalty_matrix, axis=1)

    return (
        np_penalty_matrix,
        unweighted_penalty_violation_score,
        number_constraint_violations,
    )




def compute_cost_values_P3O(policy, sample_batch, other_agent_batches, episode):
    """
    Equivalent to compute_penalty_values
    :param policy:
    :param sample_batch:
    :param other_agent_batches:
    :param episode:
    :return:
    """

    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[
            SampleBatch.REWARDS
        ]

    (
        np_penalty_matrix,
        np_unweighted_penalty_violation_score,
        amount_constraint_violations,
    ) = calculate_penalty_score_P3O(sample_batch=sample_batch, policy=policy)

    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = (
        np_unweighted_penalty_violation_score
    )
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = (
        amount_constraint_violations
    )
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (
        amount_constraint_violations > 0
    ).astype(int)

    sample_batch[SampleBatch_Custom.COST_OBS] = np_penalty_matrix
    
    for idx, column in enumerate(np.transpose(np_penalty_matrix).tolist()):
        sample_batch[f"{SampleBatch_Custom.COST_OBS}{idx}"] = np.array(column)
    
    for i in range(policy.model.amount_constraints):
        sample_batch["J_cost_"+str(i)] = np_penalty_matrix[:, i].sum(keepdims=True).repeat(np_penalty_matrix.shape[0])
    
    return sample_batch


## P3O ##


## Lagrange ##
def compute_constrained_penalized_rewards_lagrange(policy, sample_batch, other_agent_batches, episode):

    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_lambda_weighted_penalty_score, np_unweighted_penalty_violation_score, amount_constraint_violations = \
        calculate_penalty_score_lagrange(sample_batch=sample_batch,
                                               policy=policy)
    sample_batch[Postprocessing_Custom.LAMBDA_WEIGHTED_CONSTRAINT_PENALTIES] = np_lambda_weighted_penalty_score
    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations>0).astype(int)

    # Updating reward, including penalty
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] - sample_batch[Postprocessing_Custom.LAMBDA_WEIGHTED_CONSTRAINT_PENALTIES]

    return sample_batch

def calculate_penalty_score_lagrange(policy, sample_batch):
    lambda_model = policy.model.lambda_penalty_model
    
    np_penalty_matrix = calculate_constraint_violations(policy.A, policy.b, sample_batch[SampleBatch.ACTIONS])

    torch_penalty_model_input = torch.from_numpy(np_penalty_matrix).float().to(lambda_model.availabe_device)

    torch_penalty_scores = lambda_model(torch_penalty_model_input)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=sample_batch[SampleBatch.ACTIONS])

    lambda_weighted_penalty_score = torch_penalty_scores.cpu().detach().numpy().flatten()
    # for analysis purposes we also want the strength (aka sum) of the penalty violations, i.e. only values >= 0
    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)
    return lambda_weighted_penalty_score, unweighted_penalty_violation_score, amount_constraint_violations

### Lagrange ##

### IPO ####
def calculate_penalty_score_ipo(policy, sample_batch):        
    np_penalty_matrix = calculate_constraint_violations(policy.A, policy.b, sample_batch[SampleBatch.ACTIONS])

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=sample_batch[SampleBatch.ACTIONS])

    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)

    np_penalty_matrix = np.maximum(np_penalty_matrix, 0) #set all negative values to 0
    
    np_penalty_matrix -= 0.001 #TODO do not harcode this value this is because the log barrier function is not defined for 0 so we need to allow a small value
    
    np_processed_penalty_values = np.where(np_penalty_matrix < 0, np.log(-np_penalty_matrix), 1000.0) #use a high value for the log of 0 #TODO do not hardcode this value
    
    t_weight = policy.t_weight

    np_processed_penalty_values = np_processed_penalty_values / t_weight

    np_weighted_processed_penalty_value = np.sum(np_processed_penalty_values, axis=1) #summing up all penalties


    return np_weighted_processed_penalty_value, unweighted_penalty_violation_score, amount_constraint_violations


def compute_constraint_penalized_rewards_ipo(policy, sample_batch, other_agent_batches, episode):
    """
    NOT USED ANY MORE
    :param policy:
    :param sample_batch:
    :param other_agent_batches:
    :param episode:
    :return:
    """
    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_weighted_processed_penalty_value, np_unweighted_penalty_violation_score, amount_constraint_violations\
        = calculate_penalty_score_ipo(
        sample_batch=sample_batch,
        policy=policy)

    sample_batch[Postprocessing_Custom.LOG_BARRIER_CONSTRAINT_PENALTIES] = np_weighted_processed_penalty_value
    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations > 0).astype(int)

    # Overwriting reward
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] - sample_batch[Postprocessing_Custom.LOG_BARRIER_CONSTRAINT_PENALTIES]

    return sample_batch
### IPO ####


def compute_costs_cpo(policy, sample_batch, other_agent_batches, episode, bootstrap_costs):
    np_penalty_matrix = calculate_constraint_violations(policy.A, policy.b, sample_batch[SampleBatch.ACTIONS])

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=sample_batch[SampleBatch.ACTIONS])

    sample_batch[SampleBatch_Custom.COST_OBS] = np_penalty_matrix.sum(axis=1)
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations > 0).astype(int)
    
    if sample_batch[SampleBatch.DONES][-1] or not bootstrap_costs:
        last_r = 0.0
    else:
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        last_r = policy._cost(**input_dict)
    
    costs_per_steps = np.concatenate([sample_batch[SampleBatch_Custom.COST_OBS], np.array([last_r])])
    
    gamma = policy.config["gamma"]
    
    gammas = gamma ** np.arange(len(costs_per_steps)) 
    
    discounted_costs = np.sum(costs_per_steps * gammas)
    sample_batch["EPSIODE_COSTS"] = discounted_costs[None].repeat(sample_batch[SampleBatch_Custom.COST_OBS].shape[0])
    
    
    return sample_batch
    

def compute_cost_advantages(
    rollout: SampleBatch,
    last_cost_val: float,
    index_cost_constraint: int,
    gamma: float = 0.9,
    lambda_: float = 1.0,
    cost_vf_use_gae: bool = True,
    cost_vf_use_critic: bool = True,
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
    # f'{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}' <- SampleBatch.VF_PREDS
    # last_cost_val <- last_r
    # f'{SampleBatch_Custom.COST_OBS}{index_cost_constraint}' <- SampleBatch.REWARDS
    # f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}' <- Postprocessing.ADVANTAGES
    # f'{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}' <- Postprocessing.VALUE_TARGETS
    # cost_vf_use_critic <- use_critic
    # cost_vf_use_gae <- use_gae
    assert (
        f"{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}" in rollout
        or not cost_vf_use_critic
    ), "cost_vf_use_critic=True but values not found"
    assert (
        cost_vf_use_critic or not cost_vf_use_gae
    ), "Can't use gae without using a value function"

    if cost_vf_use_gae:
        vpred_t = np.concatenate(
            [
                rollout[f"{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}"],
                np.array([last_cost_val]),
            ]
        )
        delta_t = (
            rollout[f"{SampleBatch_Custom.COST_OBS}{index_cost_constraint}"]
            + gamma * vpred_t[1:]
            - vpred_t[:-1]
        )
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[f"{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}"] = (
            discount_cumsum(delta_t, gamma * lambda_)
        )
        rollout[
            f"{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}"
        ] = (
            rollout[f"{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}"]
            + rollout[f"{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}"]
        ).astype(
            np.float32
        )
    else:
        rewards_plus_v = np.concatenate(
            [
                rollout[f"{SampleBatch_Custom.COST_OBS}{index_cost_constraint}"],
                np.array([last_cost_val]),
            ]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if cost_vf_use_critic:
            rollout[
                f"{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}"
            ] = (
                discounted_returns
                - rollout[f"{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}"]
            )
            rollout[
                f"{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}"
            ] = discounted_returns
        else:
            rollout[
                f"{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}"
            ] = discounted_returns
            rollout[
                f"{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}"
            ] = np.zeros_like(
                rollout[
                    f"{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}"
                ]
            )

    rollout[f"{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}"] = (
        rollout[
            f"{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}"
        ].astype(np.float32)
    )

    return rollout


def compute_cost_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory for the COSTS

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
    for idx_cost in range(policy.model.amount_constraints):
        # Trajectory is actually complete -> last r=0.0.
        if sample_batch[SampleBatch.DONES][-1]:
            last_cost_val = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            # Create an input dict according to the Model's requirements.
            input_dict = sample_batch.get_single_step_input_dict(
                policy.model.view_requirements, index="last"
            )
            last_cost_val = policy._cost_value(idx_cost=idx_cost, **input_dict)

        # Adds the policy logits, VF preds, and advantages to the batch,
        # using GAE ("generalized advantage estimation") or not.
        batch = compute_cost_advantages(
            rollout=sample_batch,
            last_cost_val=last_cost_val,
            index_cost_constraint=idx_cost,
            gamma=policy.config["gamma"],
            lambda_=policy.config["lambda"],
            cost_vf_use_gae=policy.config["model"]["custom_model_config"][
                "cost_vf_use_gae"
            ],
            cost_vf_use_critic=policy.config["model"]["custom_model_config"].get(
                "cost_vf_use_critic", True
            ),
        )

    return batch



def compute_advantages_single(
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
        "VF_PREDS_COSTS" in rollout or not use_critic
    ), "use_critic=True but values not found"
    assert use_critic or not use_gae, "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate([rollout["VF_PREDS_COSTS"], np.array([last_r])])
        delta_t = rollout[SampleBatch_Custom.COST_OBS] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout["ADVANTAGES_COSTS"] = discount_cumsum(delta_t, gamma * lambda_)
        rollout["VALUE_TARGETS_COSTS"] = (
            rollout["ADVANTAGES_COSTS"] + rollout["VF_PREDS_COSTS"]
        ).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch_Custom.COST_OBS], np.array([last_r])]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if use_critic:
            rollout["ADVANTAGES_COSTS"] = (
                discounted_returns - rollout["VF_PREDS_COSTS"]
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

def compute_gae_for_sample_batch_single(
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
        last_r = policy._cost(**input_dict)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages_single(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
    )

    return batch

