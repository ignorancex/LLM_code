import logging

import cooper
import torch.nn.functional as F

from ..models.sparse import layers

logger = logging.getLogger(__name__)


class ConstrainedProblem(cooper.ConstrainedMinimizationProblem):
    def __init__(self, weight_decay: float, target_sparsity: float):
        super().__init__()

        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        self.weight_decay = weight_decay

        logger.info(f"Setting up ConstrainedL0Problem with target_sparsity={target_sparsity:.2e}")

        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1)

        self.constraint = cooper.Constraint(constraint_type=cooper.ConstraintType.INEQUALITY, multiplier=multiplier)

        assert 0 <= target_sparsity <= 1
        self.target_density = 1 - target_sparsity

    def compute_sparsity_stats(self, model, is_test_time: bool = False):
        model_stats = layers.get_model_stats(model, is_test_time=is_test_time)
        model_density = model_stats.compute_density_stats()[1]

        squared_l2_norm = model_stats.sq_l2_norm

        return model_density, squared_l2_norm

    def compute_cmp_state(self, model, inputs, targets) -> cooper.CMPState:
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")

        model_density, squared_l2_norm = self.compute_sparsity_stats(model, is_test_time=False)

        if self.weight_decay != 0:
            # model_stats.sq_l2_norm is the (expected) squared l2 norm of the weights
            # and biases
            loss += 0.5 * squared_l2_norm * self.weight_decay

        # Computing the violation of the sparsity constraint
        sparsity_violation = model_density - self.target_density

        constraint_state = cooper.ConstraintState(violation=sparsity_violation)

        batch_log_metrics = dict(
            model_density=model_density.detach(), squared_l2_norm=squared_l2_norm.detach(), logits=logits.detach()
        )

        return cooper.CMPState(
            loss=loss, observed_constraints={self.constraint: constraint_state}, misc=batch_log_metrics
        )


class PenalizedProblem(cooper.ConstrainedMinimizationProblem):
    def __init__(self, weight_decay: float, penalty_lambda: float):
        super().__init__()

        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        self.weight_decay = weight_decay

        logger.info(f"Setting up PenalizedL0Problem problem with penalty_lambda={penalty_lambda:.2e}")
        self.penalty_lambda = penalty_lambda

    def compute_sparsity_stats(self, model, is_test_time: bool = False):
        model_stats = layers.get_model_stats(model, is_test_time=is_test_time)
        model_density = model_stats.compute_density_stats()[1]

        squared_l2_norm = model_stats.sq_l2_norm

        return model_density, squared_l2_norm

    def compute_cmp_state(self, model, inputs, targets) -> cooper.CMPState:
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")

        model_density, squared_l2_norm = self.compute_sparsity_stats(model, is_test_time=False)

        if self.weight_decay != 0:
            # model_stats.sq_l2_norm is the (expected) squared l2 norm of the weights
            # and biases
            loss += 0.5 * squared_l2_norm * self.weight_decay

        # Add the penalty term to the loss
        loss += self.penalty_lambda * model_density

        batch_log_metrics = dict(
            model_density=model_density.detach(), squared_l2_norm=squared_l2_norm.detach(), logits=logits.detach()
        )

        return cooper.CMPState(loss=loss, misc=batch_log_metrics)
