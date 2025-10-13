import numpy as np
import scipy

from .matrix_operations import (
    multiply_each_row_of_the_matrix_by_the_nth_vector_entry,
    sum_up_dyadic_product_of_row_vectors,
)
from .spline_evaluator import (
    PhysicalSpline,
    eval_all_base_all_time,
)


def generate_heading_regularisation_matrix(
    base_function: PhysicalSpline, time: np.ndarray, heading: np.ndarray, weight_vec: np.ndarray
) -> np.ndarray:
    """Generates the regularisation matrix that prefers solutions that have at time[i] the heading[i]

        Args:
            base_function (_type_): Spline model
            time (np.ndarray): time vector must have the same length as heading vector
            heading (np.ndarray): heading vector must have the same length as the time vector
            weight_vec (np.ndarray): weighing each heading point individually.
            Must have the same length as time and heading vector.

    Returns:
        np.array: regularisation matrix that needs to be added to the optimization problem
    """
    derivative = 1
    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, derivative)
    """
    Q = np.zeros((len(base_function.weights) * 2, len(base_function.weights) * 2))
    for j in range(len(time)):
        base_function_all_at_t_np = base_function_all_at_t_np_list[j]
        tan_heading = np.sin(heading[j]) / np.cos(heading[j]) if (np.cos(heading[j]) != 0) else np.inf
        cotan_heading = np.cos(heading[j]) / np.sin(heading[j]) if (np.sin(heading[j]) != 0) else np.inf
        if np.abs(tan_heading) < np.abs(cotan_heading):
            c_dyadic = np.block([base_function_all_at_t_np * tan_heading, -base_function_all_at_t_np])
        else:
            c_dyadic = np.block([-base_function_all_at_t_np, base_function_all_at_t_np * cotan_heading])
        Q += weight_vec[j] * np.outer(np.transpose(c_dyadic), c_dyadic)
    """

    weight_vec_sqrt = np.sqrt(weight_vec)

    # suppress warnings about devision with zero, since np.where can deal with `inf` here
    with np.errstate(divide="ignore"):
        weight_dim1 = (
            np.where(np.abs(np.cos(heading)) >= np.abs(np.sin(heading)), np.sin(heading) / np.cos(heading), -1)
            * weight_vec_sqrt
        )
        weight_dim2 = (
            np.where(np.abs(np.sin(heading)) > np.abs(np.cos(heading)), np.cos(heading) / np.sin(heading), -1)
            * weight_vec_sqrt
        )
    base_function_all_weighted_dim1 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, weight_dim1
    )
    base_function_all_weighted_dim2 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, weight_dim2
    )
    base_function_all_weighted_all_dim = np.block([base_function_all_weighted_dim1, base_function_all_weighted_dim2])
    return sum_up_dyadic_product_of_row_vectors(base_function_all_weighted_all_dim)


def generate_yaw_rate_heading_matrix(
    base_function: PhysicalSpline,
    time: np.ndarray,
    heading: np.ndarray,
    yaw_rate: np.ndarray,
    weight_vec: np.ndarray,
) -> np.ndarray:
    """Generates the regularisation matrix that prefers solutions that have at time[i] the heading[i]
    and yaw_rate[i] (derivative of heading reg)

        Args:
            base_function (_type_): Spline model
            time (np.ndarray): time vector must have the same length as heading vector
            heading (np.ndarray): heading vector must have the same length as the time vector
            yaw_rate (np.ndarray): heading vector must have the same length as the time vector
            weight_vec (np.ndarray): weighing each heading point individually.
            Must have the same length as time and heading vector.

    Returns:
        np.array: regularisation matrix that needs to be added to the optimization problem
    """
    base_function_all_at_t_np_list_dot = eval_all_base_all_time(base_function, time, 1)
    base_function_all_at_t_np_list_dot_dot = eval_all_base_all_time(base_function, time, 2)

    sin_heading_np = np.sin(heading)
    cos_heading_np = np.cos(heading)
    base_function_dyadic_dim_1 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list_dot_dot, sin_heading_np * np.sqrt(weight_vec)
    ) + multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list_dot, cos_heading_np * yaw_rate * np.sqrt(weight_vec)
    )
    base_function_dyadic_dim_2 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list_dot_dot, -cos_heading_np * np.sqrt(weight_vec)
    ) + multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list_dot, sin_heading_np * yaw_rate * np.sqrt(weight_vec)
    )
    base_function_all_weighted_all_dim = np.block([base_function_dyadic_dim_1, base_function_dyadic_dim_2])
    Q = sum_up_dyadic_product_of_row_vectors(base_function_all_weighted_all_dim)
    """
    Q = np.zeros((len(base_function.weights) * 2, len(base_function.weights) * 2))
    for j in range(len(time)):
        base_function_all_dot = base_function_all_at_t_np_list_dot[j]
        base_function_all_dot_dot = base_function_all_at_t_np_list_dot_dot[j]
        sin_heading = np.sin(heading[j])
        cos_heading = np.cos(heading[j])
        c_dyadic_x = base_function_all_dot_dot * sin_heading + base_function_all_dot * cos_heading * yaw_rate[j]
        c_dyadic_y = -base_function_all_dot_dot * cos_heading + base_function_all_dot * sin_heading * yaw_rate[j]
        c_dyadic = np.block([c_dyadic_x, c_dyadic_y])
        Q += weight_vec[j] * np.outer(np.transpose(c_dyadic), c_dyadic)
    """
    return Q


def generate_simple_regularisation_matrix(
    base_function: PhysicalSpline, order: int, weight_vec: np.ndarray, exclude_first_n_coeffs: int = 2
) -> np.ndarray:
    """Generates the matrix for the coefficient regularisation that needs to be added
    to the optimization problem
        order = -1: means small sums of coeffs for the physical spline small velocities
        order = 0: means small coefficients are prefered. In the physicall spline that
        means small accelerations
        order = 1: means small coefficient differences are prefered In the physicall spline
        that means small acceleration changes
        order = 2: means small changes of the changes are prefered
        order = 3: means small changes of the (changes of the changes) are prefered

    Args:
        base_function (_type_): Spline model
        order (int): order implemented for -1, 0, 1, 2 and 3.
        weight_vec: weigh smoothing over the intervall differently (only for orders -1, 0, 1 implemented)
        exclude_first_n_coeffs (int): Excludes the first two coeffs of the physical spline,
        because they represent initial position and velocity

    Raises:
        Exception: wrong order requested

    Returns:
        np.array: regularisation matrix that needs to be added to the optimization problem
    """
    if order == -1:
        exclude_first_n_coeffs -= 1
        c_dyadic = np.zeros(len(base_function.weights) - exclude_first_n_coeffs)
        c_dyadic[0] = 1
        Q = weight_vec[0] * np.outer(np.transpose(c_dyadic), c_dyadic)
        c_dyadic[1] = (base_function.t_knot[-1] - base_function.t_knot[-2]) / 2
        c_dyadic[2] = (base_function.t_knot[-1] - base_function.t_knot[-2]) / 2
        Q += weight_vec[1] * np.outer(np.transpose(c_dyadic), c_dyadic)
        for i in range(3, len(c_dyadic)):
            c_dyadic[i - 1] += (base_function.t_knot[-1] - base_function.t_knot[-2]) / 2
            c_dyadic[i] += (base_function.t_knot[-1] - base_function.t_knot[-2]) / 2
            Q += weight_vec[i - 1] * np.outer(np.transpose(c_dyadic), c_dyadic)

    elif order == 0:
        Q = np.diag(weight_vec)
    elif order == 1:
        c_dyadic = np.zeros(len(base_function.weights) - exclude_first_n_coeffs)
        c_dyadic[0] = 1
        c_dyadic[1] = -1
        Q = weight_vec[0] * np.outer(np.transpose(c_dyadic), c_dyadic)
        for i in range(1, len(c_dyadic) - 1):
            c_dyadic[i - 1] = 0
            c_dyadic[i] = 1
            c_dyadic[i + 1] = -1
            Q += weight_vec[i] * np.outer(np.transpose(c_dyadic), c_dyadic)

    elif order == 2:
        if len(base_function.weights) - exclude_first_n_coeffs < 6:
            return np.zeros((len(base_function.weights), len(base_function.weights)))
        Q = scipy.sparse.diags(
            [1, -4, 6, -4, 1],
            [-2, -1, 0, 1, 2],
            shape=(len(base_function.weights) - 2, len(base_function.weights) - exclude_first_n_coeffs),
        ).toarray()
        Q[0, :3] = [1, -2, 1]
        Q[1, :4] = [-2, 5, -4, 1]
        Q[-1, -3:] = [1, -2, 1]
        Q[-2, -4:] = [-2, 5, -4, 1]
    elif order == 3:
        if len(base_function.weights) - exclude_first_n_coeffs < 8:
            return np.zeros((len(base_function.weights), len(base_function.weights)))
        Q = scipy.sparse.diags(
            [-1, 6, -15, 20, -15, 6, -1],
            [-3, -2, -1, 0, 1, 2, 3],
            shape=(len(base_function.weights) - 2, len(base_function.weights) - exclude_first_n_coeffs),
        ).toarray()
        Q[0, :4] = [1, -3, 3, -1]
        Q[1, :5] = [-3, 10, -12, 6, -1]
        Q[2, :6] = [3, -12, 19, -15, 6, -1]

        Q[-1, -4:] = [1, -3, 3, -1]
        Q[-2, -5:] = [-3, 10, -12, 6, -1]
        Q[-3, -6:] = [3, -12, 19, -15, 6, -1]
    else:
        raise Exception("Not Implemented order requested, only orders -1, 0, 1, 2 and 3 are implemented")
    if exclude_first_n_coeffs > 0:
        Q = scipy.linalg.block_diag(np.zeros((exclude_first_n_coeffs, exclude_first_n_coeffs)), Q)
    return Q
