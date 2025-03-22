#!/usr/bin/env python3

r"""This code is adapted from the paper: `Triangulation candidates for Bayesian
optimization` by R.Gramacy, A.Sauer and N.Wycoff.

Link to paper: https://arxiv.org/abs/2112.07457
Link to code: https://bitbucket.org/gramacylab/tricands/

This code was written using scipy routines, which leverages the C library Qhull.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def triangle_candidates_interior(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute the interior triangle candidates.

    Args:
        X: A `N x d`-dim numpy array containing the training inputs.

    Returns
        candidates: A `num_interior candidates x d`-dim numpy array containing the
            interior candidates.
        triangles: A `num_interior candidates x (d + 1)`-dim numpy array containing
            the triangle coordinates.
    """
    d = X.shape[1]
    N = X.shape[0]
    if N < d + 1:
        raise Exception("Must have N >= d + 1")

    # Find the middle of the triangles.
    triangles = Delaunay(X, qhull_options="Q12").vertices
    num_triangles = triangles.shape[0]
    candidates = np.zeros([num_triangles, d])

    for i in range(num_triangles):
        candidates[i, :] = np.mean(X[triangles[i, :], :], axis=0)

    return candidates, triangles


def triangle_candidates_fringe(X: np.ndarray) -> np.ndarray:
    r"""Compute the fringe triangle candidates.

    This method implicitly assumes that `X` lies in the hypercube [0, 1]^d.

    Args:
        X: A `N x d`-dim numpy array containing the training inputs.

    Returns
        fringe: A `num_fringe_candidates x d`-dim numpy array containing the fringe
            candidates.
    """
    d = X.shape[1]
    N = X.shape[0]
    if N < d + 1:
        raise Exception("Must have N >= d + 1")

    # Get midpoints of external (convex hull) facets and normal vectors.
    qhull = ConvexHull(X)
    num_simplices = qhull.simplices.shape[0]

    norms = np.zeros((num_simplices, d))
    boundaries = np.zeros((num_simplices, d))

    for i in range(num_simplices):
        boundaries[i, :] = np.mean(X[qhull.simplices[i, :], :], axis=0)
        norms[i, :] = qhull.equations[i, 0:d]
    # Norms off of the boundary points to get fringe candidates half-way from the
    # facet midpoints to the boundary.
    eps = np.sqrt(np.finfo(float).eps)
    ai = np.zeros([num_simplices, d])
    pos = norms > 0
    ai[pos] = (1 - boundaries[pos]) / norms[pos]
    ai[np.logical_not(pos)] = (
        -boundaries[np.logical_not(pos)] / norms[np.logical_not(pos)]
    )
    ai[np.abs(norms) < eps] < -np.inf
    alpha = np.min(ai, axis=1)

    # Half-way to the edge.
    fringe = boundaries + norms * alpha[:, np.newaxis] / 2

    return fringe


def triangle_candidates(
    X: np.ndarray,
    fringe: bool = True,
    max_num_candidates: Optional[int] = None,
    best_indices: Optional[List[int]] = None,
) -> np.ndarray:
    r"""Compute the triangle candidates.

    Args:
        X: A `N x d`-dim numpy array containing the training inputs.
        fringe: If true we compute the fringe candidates.
        max_num_candidates: The maximum number of candidates, defaults to `100 * N`.
        best_indices: A list containing the indices of the points in `X` that are
            given priority when the number of generated candidates surpasses
            `max_num_candidates`.

    Returns
        candidates: A `num_candidates x d`-dim numpy array containing the candidates.
    """
    d = X.shape[1]
    N = X.shape[0]
    if max_num_candidates is None:
        max_num_candidates = 100 * N
    if N < d + 1:
        raise Exception("Must have N >= d + 1")

    interior_candidates, triangles = triangle_candidates_interior(X)

    # Calculate midpoints of convex hull vectors.
    if fringe:
        fringe_candidates = triangle_candidates_fringe(X)
        candidates = np.concatenate([interior_candidates, fringe_candidates])
    else:
        candidates = interior_candidates

    num_generated_candidates = candidates.shape[0]
    all_indices = np.array(list(range(num_generated_candidates)))

    # Truncate the set of `all_indices` if necessary.
    if max_num_candidates < num_generated_candidates:
        adjacent_indices = np.array([])
        # First populate the `adjacent_indices` with the indices of the points
        # closest to the points highlighted by `best_indices`.
        if best_indices is not None:
            for i in range(len(best_indices)):
                adj_i = np.where(
                    np.apply_along_axis(
                        lambda x: np.any(x == best_indices[i]), 1, triangles
                    )
                )[0]
                adjacent_indices = np.unique(np.concatenate([adjacent_indices, adj_i]))
            adjacent_indices = adjacent_indices.astype(int)

        # If the set of `adjacent_indices` is still too large, truncate it randomly.
        # If the set of `adjacent_indices` is still too small, add points randomly.
        if len(adjacent_indices) >= max_num_candidates:
            selected_indices = np.random.choice(
                adjacent_indices, max_num_candidates, replace=False
            )
        else:
            if len(adjacent_indices) > 0:
                remaining_indices = np.delete(all_indices, adjacent_indices, 0)
            else:
                remaining_indices = all_indices

            other_indices = np.random.choice(
                remaining_indices,
                max_num_candidates - len(adjacent_indices),
                replace=False,
            )
            selected_indices = np.concatenate([adjacent_indices, other_indices]).astype(
                int
            )
    else:
        selected_indices = all_indices

    return candidates[selected_indices, :]
