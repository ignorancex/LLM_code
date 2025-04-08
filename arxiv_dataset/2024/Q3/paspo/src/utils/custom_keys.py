""" Module which holds key string values ENUM-style """


class Postprocessing_Custom:
    """
    Class to hold ENUM-style variables
    """

    ALLOCATION = "allocation"  # just used as tmp
    AMOUNT_ALLOCATION_VIOLATIONS = "amount_allocation_violations"
    REWARD_BASE_NO_PENALTIES = "rewards_base_no_penalties"
    LAMBDA_WEIGHTED_CONSTRAINT_PENALTIES = "lambda_weighted_constraint_penalties"
    UNWEIGHTED_PENALTY_VIOLATION_SCORE = "unweighted_penalty_violation_score"
    REWARD_INCL_CONSTRAINT_PENALTIES = "rewards_incl_constraint_penalties"
    LOG_BARRIER_CONSTRAINT_PENALTIES = "log_barrier_constraint_penalties"
    COST_ADVANTAGES = "cost_advantages_"
    COST_VALUE_TARGETS = "cost_value_targets_"

    AMOUNT_CONSTRAINT_VIOLATIONS = "amount_constraint_violations"

    BOOL_ANY_CONSTRAINT_VIOLATIONS = "bool_any_constraint_violations"

    ACTION_VIOLATIONS = "action_violation_samples"
    ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS = (
        "action_violation_constraint_index_samples"
    )

    PROCESSED_ACTIONS = "processed_actions"


class SampleBatch_Custom():
    """
    Class to hold ENUM-style variables
    """

    COST_VF_PREDS = "cost_vf_pred_"  # this is used in combination with index numbers, i.e. f'{COST_VF_PREDS_}{idx}'
    COST_OBS = "cost_obs_"
