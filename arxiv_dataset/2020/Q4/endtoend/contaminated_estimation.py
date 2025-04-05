from math import exp, floor

import numpy as np
import torch

from project_typing import *


def _get_feature_values(people: List[Agent]) -> Dict[Feature, Set[Value]]:
    """Extract the feature-value pairs from a list of agents."""
    feature_values: Dict[Feature, Set[Value]] = {}
    for person in people:
        for feature in person:
            if feature not in feature_values:
                feature_values[feature] = set()
            feature_values[feature].add(person[feature])
    return feature_values


def contaminated_estimation(pool: List[Agent], background: List[Agent], pi: float,
                            background_weights: Optional[List[float]] = None) \
        -> Tuple[float, Dict[FeatureValue, float]]:
    """Compute the β_{f,v} and β₀ for the given pool and background data.

    Args:
        pool: the uniform sample of positive-labeled agents
        background: the uniform sample of the population overall, with unknown labels
        background_weights: if present, individuals in the background sample are weighted, as needed for ESS data
    Returns:
        Tuple (β₀, {(f,v): β_{f,v}})
    """
    if background_weights is None:
        background_weights = [1. for _ in background]
    n_u = len(background)
    n_p = len(pool)
    n = n_u + n_p

    n_u_weighted = sum(background_weights)

    feature_values = _get_feature_values(pool + background)

    fv_to_index: Dict[FeatureValue, int] = {}
    index_to_fv: List[FeatureValue] = []
    for feature in feature_values:
        for value in feature_values[feature]:
            index = len(index_to_fv)
            index_to_fv.append((feature, value))
            fv_to_index[(feature, value)] = index
    num_fv = len(index_to_fv)

    x_rows: List[List[float]] = []
    s_vals: List[float] = []
    for person in pool:
        s_vals.append(1.)
        new_row = [0. for _ in range(num_fv + 1)]
        new_row[-1] = 1.
        for feature in person:
            new_row[fv_to_index[(feature, person[feature])]] = 1.
        x_rows.append(new_row)
    for person in background:
        s_vals.append(0.)
        new_row = [0. for _ in range(num_fv + 1)]
        new_row[-1] = 1.
        for feature in person:
            new_row[fv_to_index[(feature, person[feature])]] = 1.
        x_rows.append(new_row)

    s = np.array(s_vals)
    assert s.shape == (n,)
    X_np = np.array(x_rows)
    assert X_np.shape == (n, num_fv + 1)

    zX = torch.tensor(np.transpose(s) @ X_np, dtype=torch.double)
    X = torch.tensor(X_np, dtype=torch.double)
    offset = torch.full((n,), pi * n_u_weighted / n_p, dtype=torch.double)
    weights = torch.tensor([1. for _ in pool] + background_weights, dtype=torch.double)

    theta = torch.full((num_fv + 1,), -0.3, requires_grad=True, dtype=torch.double)

    for i in range(100000):
        f1 = torch.dot(zX, theta)
        f2 = torch.dot(weights, torch.log(torch.exp(torch.mv(X, theta)) + offset))
        objective = f1 - f2
        objective.backward()
        with torch.no_grad():
            if i % 1000 == 0:
                print(f"In iteration {i}, log-likelihood is {float(objective)}.")
            theta += .00001 * theta.grad.data
            theta.clamp_(max=0)
        theta.grad.data.zero_()

    baseline = exp(theta[-1])
    value_multipliers = {}
    for feature in feature_values:
        max_multiplier = max(exp(theta[fv_to_index[(feature, value)]]) for value in feature_values[feature])
        for value in feature_values[feature]:
            value_multipliers[(feature, value)] = exp(theta[fv_to_index[(feature, value)]]) / max_multiplier
        baseline *= max_multiplier
    return baseline, value_multipliers


def from_file(file_name: str) -> List[Agent]:
    entries = []
    with open(file_name, "r") as file:
        categories = file.readline().strip().split(",")
        for line in file:
            entry = {Feature(cat): Value(attr) for cat, attr in zip(categories, line.strip().split(","))}
            entries.append(entry)
    return entries


def calculate_qis(people: List[Agent], baseline: float, value_multipliers: Dict[FeatureValue, float]) -> List[float]:
    """Calculate the participation probabilities qᵢ by multiplying, for each agent, β₀ with all β_{f,v} applicable to
    this agent.
    """
    qis = []
    for person in people:
        qi = baseline
        for feature, value in person.items():
            qi *= value_multipliers[(feature, value)]
        qis.append(qi)
    return qis


def calculate_pis(people: List[Agent], qis: List[float], k: int) -> Tuple[List[float], int]:
    """Calculate the marginal selection probabilities π_{i, P}.

    First, π_{i, P} is set to (k / qᵢ) / Σ_{i∈P} 1/qᵢ. If some of these values are at least 1, the algorithm is
    repeated by reducing these values to 1, proportionally scaling the smaller values, and repeating.

    Returns:
        Tuple ([π_{i,P} for i in range(len(people))], number of clipped variables).
        If the number of clipped variables is 0, no clipping happened.
    """
    assert len(people) == len(qis)
    n = len(people)

    ais = [1 / qi for qi in qis]
    deterministic = [False for _ in range(n)]

    while True:
        corrected_k = k - sum(deterministic)
        sum_ai = sum(ai for ai, det in zip(ais, deterministic) if not det)
        pis = [1. if det else corrected_k * ai / sum_ai for ai, det in zip(ais, deterministic)]
        old_det_num = sum(deterministic)
        deterministic = [pi >= 1. for pi in pis]
        if sum(deterministic) == old_det_num:
            return pis, old_det_num


def get_pool() -> List[Agent]:
    """Get the agents in the pool and π, the fraction of positive labels in the population."""
    entries = from_file("data/UKrespondents.csv")
    for entry in entries:
        delete_attributes = ["nationbuilder_id", "primary_address1", "zip_edited", "first_name", "last_name", "email",
                             "mobile_number", "primary_address2", "primary_city", "primary_zip", "age"]
        for attr in delete_attributes:
            del entry[attr]

    return entries


def get_background() -> Tuple[List[Agent], List[float]]:
    """Get the agents from the background sample, along with their weights (in the same order)."""
    entries = from_file("data/cleaned_ESS_data_missingsdropped.csv")
    weights = []
    for entry in entries:
        del entry[Feature("idno")]
        weights.append(float(entry[Feature("weight")]))
        del entry[Feature("weight")]
    return entries, weights
