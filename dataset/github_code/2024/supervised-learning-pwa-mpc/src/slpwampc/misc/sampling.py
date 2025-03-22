import numpy as np
from scipy.optimize import linprog


def random_sample_region(
    A: np.ndarray, b: np.ndarray, n: int, np_random: np.random.Generator
) -> np.ndarray:
    """Sample n points randomly from the region defined by Ax <= b.

    Parameters
    ----------
    A : np.ndarray
        The A matrix in Ax <= b.
    b : np.ndarray
        The b matrix in Ax <= b.
    n : int
        The number of points to sample.
    np_random : np.random.Generator
        The random number generator.

    Returns
    -------
    np.ndarray
        The sampled points of shape (n, A.shape[1], 1)."""
    bounds = outer_bounding_box(A, b)
    num_succesful_points = 0
    points = []
    while num_succesful_points < n:
        point = np_random.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds], (1, A.shape[1])
        ).T
        if all(A @ point <= b):
            points.append(point)
            num_succesful_points += 1
    return np.concatenate(points, axis=1).T.reshape(-1, A.shape[1], 1)


def grid_sample_region(A: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
    """Sample points from the region defined by Ax <= b, in a uniform grid with spacing (vert and hor) d.

    Parameters
    ----------
    A : np.ndarray
        The A matrix in Ax <= b.
    b : np.ndarray
        The b matrix in Ax <= b.
    d : float
        The spacing between grid points.

    Returns
    -------
    np.ndarray
        The sampled points of shape (n, S.shape[1], 1).
    """
    bounds = outer_bounding_box(A, b)
    # generate a uniform grid within the bounds
    grids = [np.arange(b[0], b[1], d) for b in bounds]
    mesh = np.meshgrid(*grids)
    points = np.vstack([m.flatten() for m in mesh])

    # Filter points to ensure they satisfy Ax <= b
    valid_points = points[:, np.all(A @ points <= b, axis=0)].T
    return valid_points.reshape(-1, A.shape[1], 1)


def outer_bounding_box(A: np.ndarray, b: np.ndarray) -> list[tuple[float, float]]:
    """Determine the outer bounding box of the region defined by Ax <= b.

    Parameters
    ----------
    A : np.ndarray
        The A matrix in Ax <= b.
    b : np.ndarray
        The b matrix in Ax <= b.

    Returns
    -------
    list[tuple[float, float]]
        A list of tuples containing the minimum and maximum bounds for each dimension.
    """
    nx = A.shape[1]  # number of dimensions
    bounds = []

    # determine an outer bounding box for the dimensions
    for i in range(nx):
        c = np.zeros(nx)
        c[i] = 1  # Minimize or maximize x_i

        # Minimize x_i
        res_min = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
        if res_min.success:
            min_bound = res_min.x[i]
        else:
            raise ValueError(f"No feasible solution found when minimizing x[{i}]")

        # Maximize x_i
        res_max = linprog(-c, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
        if res_max.success:
            max_bound = res_max.x[i]
        else:
            raise ValueError(f"No feasible solution found when maximizing x[{i}]")

        bounds.append((min_bound, max_bound))
    return bounds


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    S = np.array([[1, 0]])
    T = np.array([[1]])
    D = np.array([[-1, 1], [-3, -1], [0.2, 1], [-1, 0], [1, 0], [0, -1]])
    E = np.array([[15], [25], [9], [6], [8], [10]])
    p = grid_sample_region(np.vstack(D), np.vstack(E), 1)
    plt.scatter(p[0, :], p[1, :])
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.show()
