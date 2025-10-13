import cooper
import torch


class UnconstrainedMixtureSeparation(cooper.ConstrainedMinimizationProblem):
    def __init__(self, lmbda_pen: float = 0.0):
        """Implements CMP for separating the MoG dataset with a linear predictor.
        Args:
            lmbda_pen: Lagrange multiplier for the penalty term. Defaults to ``0.0``.
        """
        self.lmbda_pen = lmbda_pen
        super().__init__()

    def compute_cmp_state(self, model, inputs, targets):
        logits = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten(), targets)

        probs = torch.sigmoid(logits)
        penalty = -torch.mean(1 - probs)

        penalized_loss = loss + self.lmbda_pen * penalty

        return cooper.CMPState(loss=penalized_loss)


class MixtureSeparation(cooper.ConstrainedMinimizationProblem):
    """Implements CMP for separating the MoG dataset with a linear predictor.

    Args:
        constraint_level: Minimum proportion of points to be predicted as belonging to
            the blue class. Ignored when ``is_constrained==False``. Defaults to ``0.7``.
    """

    def __init__(self, constraint_level: float = 0.7):
        super().__init__()

        constraint_type = cooper.ConstraintType.INEQUALITY
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1)
        self.rate_constraint = cooper.Constraint(
            constraint_type=constraint_type, formulation_type=cooper.formulations.Lagrangian, multiplier=multiplier
        )

        self.constraint_level = constraint_level

    def compute_cmp_state(self, model, inputs, targets):
        logits = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten(), targets)

        # Hinge approximation of the rate
        probs = torch.sigmoid(logits)

        # Surrogate "constraint": prob_0 >= constraint_level -> constraint_level - prob_0 <= 0
        differentiable_violation = self.constraint_level - torch.mean(1 - probs)

        # Use the true rate of blue predictions as the constraint
        classes = logits >= 0.0
        prop_0 = torch.sum(classes == 0) / targets.numel()

        # Constraint: prop_0 >= constraint_level -> constraint_level - prop_0 <= 0
        strict_violation = self.constraint_level - prop_0

        constraint_state = cooper.ConstraintState(violation=differentiable_violation, strict_violation=strict_violation)

        return cooper.CMPState(loss=loss, observed_constraints={self.rate_constraint: constraint_state})
