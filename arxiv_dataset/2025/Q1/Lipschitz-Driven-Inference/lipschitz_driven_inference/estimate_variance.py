import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
import cvxpy as cp
from sklearn.metrics.pairwise import haversine_distances


def estimate_minimum_variance(
    locations: ArrayLike,
    responses: ArrayLike,
    lipschitz_bound: float,
    data_on_sphere: bool = False,
):
    n = len(locations)
    locations = np.asarray(locations)

    # Build an n^2 x n matrix with a each possible pair of +-1 in a row
    # scatter in matrix entries
    A_upper_part = sparse.block_diag((np.ones((1, n)),) * n)
    A_lower_part = sparse.hstack((sparse.eye(n),) * n)
    A = A_upper_part - A_lower_part
    # get cost matrix
    if data_on_sphere:
        # Compute the haversine distances between the locations, convert to kilometers
        D = haversine_distances(np.radians(locations), np.radians(locations)) * 6371
    else:
        D = np.linalg.norm(locations[:, None] - locations[None], ord=2, axis=-1)
    cost = D.ravel() * lipschitz_bound
    # Use cvxpy to solve the problem
    x = cp.Variable((n))
    constraints = [A.T @ x <= cost]
    objective = cp.Minimize(cp.sum_squares(x - responses[:, 0]))
    prob = cp.Problem(objective, constraints)
    prob.solve(cp.CLARABEL)

    return prob.value / n

def fast_estimate_variance(
    locations: ArrayLike,
    responses: ArrayLike,
    lipschitz_bound: float,
    data_on_sphere: bool = False,
):
    """
    # Use Nearest Neighbors regression to estimate the variance. 
    # This is a fast heuristic that is not guaranteed to be consistent. 
    # We perform regression using all responses but the one at the current location.
    # We then compute the variance of the residuals.
    # This is just the cross-validation score of a nearest neighbor regression model.
    """
    locations = np.asarray(locations)

    # compute pairwsie distances between the locations
    if data_on_sphere:
        # Compute the haversine distances between the locations, convert to kilometers, doesn't matter
        D = haversine_distances(np.radians(locations), np.radians(locations)) * 6371
    else:
        D = np.sum(np.square(locations[:, None] - locations[None]), axis=-1)
    # Find a vector of the 1 nearest neighbors for each location, not including the location itself
    nearest_neighbor = np.argsort(D, axis=1)[:, 1]

    # Compute the residuals
    residuals = responses - responses[nearest_neighbor]
    
    # Compute the mean of the squared residuals
    return np.mean(residuals ** 2) /2 
