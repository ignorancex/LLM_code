import numpy as np
from scipy import stats


def compute_correlation(data):
    """Compute the Pearson correlation coefficient for the provided data."""
    if len(data) < 2:
        raise ValueError("Data must contain at least two pairs")

    # Convert list of tuples to a 2D numpy array
    data_array = np.array(data)

    # Ensure the array has exactly two columns
    if data_array.shape[1] != 2:
        raise ValueError("Each tuple must contain exactly two elements")

    # Compute Pearson correlation coefficient
    r, _ = stats.pearsonr(data_array[:, 0], data_array[:, 1])
    return r


def p_value_for_pcc(r, n):
    """Calculate the p-value for a given Pearson correlation coefficient."""
    df = n - 2  # degrees of freedom
    if df <= 0:
        raise ValueError("Number of samples must be greater than 2")

    t_stat = r * np.sqrt(df / (1 - r**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return p_value


def brute_force_search(data, bounds, grid_size=10):
    """Brute force search to compute the maximal change in correlation and p-value."""
    x_range = (bounds[0][0], bounds[1][0])
    y_range = (bounds[0][1], bounds[1][1])

    original_r = compute_correlation(data)
    original_p_value = p_value_for_pcc(original_r, len(data))

    max_r_diff = 0
    max_p_value_diff = 0

    # Create a grid within the boundary space
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)

    for x in x_grid:
        for y in y_grid:
            # Append the grid point
            new_data = data + [(x, y)]
            new_r = compute_correlation(new_data)
            new_p_value = p_value_for_pcc(new_r, len(new_data))

            # Compute the differences
            r_diff = abs(new_r - original_r)
            p_value_diff = abs(new_p_value - original_p_value)

            # Update max differences
            max_r_diff = max(max_r_diff, r_diff)
            max_p_value_diff = max(max_p_value_diff, p_value_diff)

    return max_r_diff, max_p_value_diff
