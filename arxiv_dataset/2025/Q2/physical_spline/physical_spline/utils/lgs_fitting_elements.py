import numpy as np

from .matrix_operations import (
    generate_1d_weighted_dyadic_product_sum,
    generate_2d_weighted_dyadic_product_and_sum_them_up,
    multiply_each_row_of_the_matrix_by_the_nth_vector_entry,
    multiply_each_row_vector_with_weight_and_sum_them_up,
    sum_up_dyadic_product_of_row_vectors,
)
from .simple_fitting import estimate_locale_heading
from .spline_evaluator import (
    PhysicalSpline,
    eval_all_base_all_time,
)


def generate_1d_point_fitting_matrix_and_vector(
    base_function: PhysicalSpline, time: np.ndarray, value: np.ndarray, derivative: int, weight_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Matrix Q and the vector b that needs to be added to the optimisation problem,
    to take the handed points with these weights into consideration

    Args:
        base_function (PhysicalSpline): Spline_model
        time (np.ndarray): time vector must have the same length as the value vector the pair is
        fcn(time[i]) = value[i]
        value (np.ndarray): value vector must have the same length as time vector
        derivative (int, optional): indicates if the values are position, velocities or accelerations.
        weight_vec (np.ndarray): Must have the same length as time and value vector. You can
        specify individual weights for each point.
    Returns :
        tuple[np.array,np.array]: Matrix Q and vector b to be added to the optimization problem
    """
    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, derivative)

    Q = generate_1d_weighted_dyadic_product_sum(weight_vec, base_function_all_at_t_np_list)
    b = multiply_each_row_vector_with_weight_and_sum_them_up(base_function_all_at_t_np_list, value * weight_vec)
    # Q = np.zeros((len(base_function.weights), len(base_function.weights)))
    # b = np.zeros(len(base_function.weights))
    # base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, derivative)
    # for j in range(len(time)):
    #    base_function_all_at_t_np = base_function_all_at_t_np_list[j]
    #    Q += weight_vec[j] * np.outer(np.transpose(base_function_all_at_t_np), base_function_all_at_t_np)
    #    b += weight_vec[j] * base_function_all_at_t_np * value[j]
    return Q, b


def generate_2d_point_fitting_matrix_and_vector(
    base_function: PhysicalSpline,
    time: np.ndarray,
    value_x: np.ndarray,
    value_y: np.ndarray,
    derivative: int,
    weight_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Matrix Q and the vector b that needs to be added to the optimisation problem, to
    take the handed points with these weights into consideration

    Args:
        base_function (PhysicalSpline): Spline_model
        time (np.ndarray): time vector must have the same length as the value_x and value_y vector
        the pair is fcn_1(time[i]) = value_x[i], fcn_2(time[i]) = value_y[i]
        value_x (np.ndarray): x value vector must have the same length as time and value_y vector
        value_y (np.ndarray): y value vector must have the same length as time and value_x vector
        derivative (int, optional): indicates if the value_x,value_y are position values or xDot,
        yDot or xDotDot, yDotDot values.
        weight_vec (np.ndarray): weighing each point individuallly must have the same
        length as value_x,value_y and time.

    Returns:
        tuple[np.array,np.array]: Matrix Q and vector b to be added to the optimization problem
    """

    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, derivative)

    Q_one_dimension = generate_1d_weighted_dyadic_product_sum(weight_vec, base_function_all_at_t_np_list)
    b_1 = multiply_each_row_vector_with_weight_and_sum_them_up(base_function_all_at_t_np_list, value_x * weight_vec)
    b_2 = multiply_each_row_vector_with_weight_and_sum_them_up(base_function_all_at_t_np_list, value_y * weight_vec)
    """
    Q_one_dimension = np.zeros((len(base_function.weights), len(base_function.weights)))
    b_1 = np.zeros(len(base_function.weights))
    b_2 = np.zeros(len(base_function.weights))
    for j in range(len(time)):
        base_function_all_at_t_np = base_function_all_at_t_np_list[j]
        Q_one_dimension += weight_vec[j] * np.outer(np.transpose(base_function_all_at_t_np), base_function_all_at_t_np)
        b_1 += weight_vec[j] * base_function_all_at_t_np * value_x[j]
        b_2 += weight_vec[j] * base_function_all_at_t_np * value_y[j]
    """
    length = len(Q_one_dimension)
    Q = np.block(
        [
            [Q_one_dimension, np.zeros([length, length])],
            [np.zeros([length, length]), Q_one_dimension],
        ]
    )
    b = np.block([b_1, b_2])
    return Q, b


def generate_lateral_point_fit_matrix_and_vector(
    base_function: PhysicalSpline,
    time: np.ndarray,
    value_x: np.ndarray,
    value_y: np.ndarray,
    heading: np.ndarray,
    derivative: int,
    weight_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Matrix Q and the vector b that needs to be added to the optimisation problem,
    to take lateral error to these points with these weights and headings into consideration

    Args:
        base_function (PhysicalSpline): Spline_model
        time (np.ndarray): time vector must have the same length as the value_x and value_y vector
        the pair is fcn_1(time[i]) = value_x[i], fcn_2(time[i]) = value_y[i]
        value_x (np.ndarray): x value vector must have the same length as time and value_y vector
        value_y (np.ndarray): y value vector must have the same length as time and value_x vector
        heading (np.ndarray): heading at each point must have the same length as value_x, value_y
        derivative (int, optional): indicates if the value_x,value_y are position values or xDot,
        yDot or xDotDot, yDotDot values.
        weight_vec (np.ndarray, optional): weighing each point individuallly must have the same
        length as value_x,value_y and time.

    Returns:
        tuple[np.array,np.array]: Matrix Q and vector b to be added to the optimization problem
    """
    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, derivative)
    base_function_all_weighted_dim1 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, -np.sin(heading) * np.sqrt(weight_vec)
    )
    base_function_all_weighted_dim2 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, np.cos(heading) * np.sqrt(weight_vec)
    )
    base_function_all_weighted_all_dim = np.block([base_function_all_weighted_dim1, base_function_all_weighted_dim2])
    Q = sum_up_dyadic_product_of_row_vectors(base_function_all_weighted_all_dim)
    b = multiply_each_row_vector_with_weight_and_sum_them_up(
        base_function_all_weighted_all_dim,
        np.sqrt(weight_vec) * (-np.sin(heading) * value_x + np.cos(heading) * value_y),
    )
    """
    Q = np.zeros((len(base_function.weights) * 2, len(base_function.weights) * 2))
    b = np.zeros(len(base_function.weights) * 2)
    for j in range(len(time)):
        base_function_all_at_t_np = base_function_all_at_t_np_list[j]
        c_dyadic = np.block(
            [
                -np.array(base_function_all_at_t_np) * np.sin(heading[j]),
                np.array(base_function_all_at_t_np) * np.cos(heading[j]),
            ]
        )
        Q += weight_vec[j] * np.outer(np.transpose(c_dyadic), c_dyadic)
        b += weight_vec[j] * (-np.sin(heading[j]) * value_x[j] + np.cos(heading[j]) * value_y[j]) * c_dyadic
    """
    return Q, b


def generate_longitudinal_point_fit_matrix_and_vector(
    base_function: PhysicalSpline,
    time: np.ndarray,
    value_x: np.ndarray,
    value_y: np.ndarray,
    heading: np.ndarray,
    derivative: int,
    weight_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Matrix Q and the vector b that needs to be added to the optimisation problem,
    to take longitudinal errors to these points with these weights and headings into consideration

    Args:
        base_function (PhysicalSpline): Spline_model
        time (np.ndarray): time vector must have the same length as the value_x and value_y vector
        the pair is fcn_1(time[i]) = value_x[i], fcn_2(time[i]) = value_y[i]
        value_x (np.ndarray): x value vector must have the same length as time and value_y vector
        value_y (np.ndarray): y value vector must have the same length as time and value_x vector
        heading (np.ndarray): heading at each point must have the same length as value_x, value_y
        derivative (int, optional): indicates if the value_x,value_y are position values or xDot,
        yDot or xDotDot, yDotDot values.
        weight_vec (np.ndarray): weighing each point individuallly must have the same
        length as value_x,value_y and time.

    Returns:
        tuple[np.array,np.array]: Matrix Q and vector b to be added to the optimization problem
    """
    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, derivative)

    base_function_all_weighted_dim1 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, np.cos(heading) * np.sqrt(weight_vec)
    )
    base_function_all_weighted_dim2 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, np.sin(heading) * np.sqrt(weight_vec)
    )
    base_function_all_weighted_all_dim = np.block([base_function_all_weighted_dim1, base_function_all_weighted_dim2])
    Q = sum_up_dyadic_product_of_row_vectors(base_function_all_weighted_all_dim)
    b = multiply_each_row_vector_with_weight_and_sum_them_up(
        base_function_all_weighted_all_dim,
        np.sqrt(weight_vec) * (np.cos(heading) * value_x + np.sin(heading) * value_y),
    )
    """
    Q = np.zeros((len(base_function.weights) * 2, len(base_function.weights) * 2))
    b = np.zeros(len(base_function.weights) * 2)
    for j in range(len(time)):
        base_function_all_at_t_np = base_function_all_at_t_np_list[j]
        c_dyadic = np.block(
            [
                np.array(base_function_all_at_t_np) * np.cos(heading[j]),
                np.array(base_function_all_at_t_np) * np.sin(heading[j]),
            ]
        )
        Q += weight_vec[j] * np.outer(np.transpose(c_dyadic), c_dyadic)
        b += weight_vec[j] * (np.cos(heading[j]) * value_x[j] + np.sin(heading[j]) * value_y[j]) * c_dyadic
    """
    return Q, b


def generate_velocity_heading_matrix_and_vector(
    base_function: PhysicalSpline,
    time: np.ndarray,
    value_x_dot: np.ndarray,
    value_y_dot: np.ndarray,
    weight_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Matrix Q and the vector b that needs to be added to the optimisation problem,
    to take cartesian velocities into account for the heading estimation
    Args:
        base_function (PhysicalSpline): Spline_model
        time (np.ndarray): time vector must have the same length as the value_x and value_y vector
        value_x_dot (np.ndarray): x_dot value vector must have the same length as time and value_y vector
        value_y_dot (np.ndarray): y_dot value vector must have the same length as time and value_y vector
        weight_vec (np.ndarray): weighing each point individuallly must have the same
        length as value_x_dot, value_y_dot and time.
    Returns:
        tuple[np.array,np.array]: Matrix Q and vector b to be added to the optimization problem
    """
    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, 0)

    Q_fit = generate_2d_weighted_dyadic_product_and_sum_them_up(
        value_x_dot * np.sqrt(weight_vec), -value_y_dot * np.sqrt(weight_vec), base_function_all_at_t_np_list
    )
    Q_one_dimension = generate_1d_weighted_dyadic_product_sum(
        weight_vec * np.sqrt(value_x_dot**2 + value_y_dot**2), base_function_all_at_t_np_list
    )
    Q = Q_fit + np.block(
        [
            [Q_one_dimension, np.zeros([len(Q_one_dimension), len(Q_one_dimension)])],
            [np.zeros([len(Q_one_dimension), len(Q_one_dimension)]), Q_one_dimension],
        ]
    )

    base_function_all_weighted_dim1_b = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, value_y_dot * weight_vec
    )
    base_function_all_weighted_dim2_b = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, value_x_dot * weight_vec
    )
    base_function_all_weighted_all_dim_b = np.block(
        [base_function_all_weighted_dim1_b, base_function_all_weighted_dim2_b]
    )
    b = multiply_each_row_vector_with_weight_and_sum_them_up(base_function_all_weighted_all_dim_b)
    """
    Q = np.zeros((len(base_function.weights) * 2, len(base_function.weights) * 2))
    b = np.zeros(len(base_function.weights) * 2)
    for j in range(len(time)):
        base_function_all_at_t_np = base_function_all_at_t_np_list[j]
        c_dyadic = np.block([value_x_dot[j] * base_function_all_at_t_np, -value_y_dot[j] * base_function_all_at_t_np])
        Q += weight_vec[j] * np.outer(np.transpose(c_dyadic), c_dyadic)
        Q_one_dimension = (
            weight_vec[j]
            * np.sqrt(value_x_dot[j] ** 2 + value_y_dot[j] ** 2)
            * np.outer(np.transpose(base_function_all_at_t_np), base_function_all_at_t_np)
        )
        Q += np.block(
            [
                [Q_one_dimension, np.zeros([len(Q_one_dimension), len(Q_one_dimension)])],
                [np.zeros([len(Q_one_dimension), len(Q_one_dimension)]), Q_one_dimension],
            ]
        )
        b += weight_vec[j] * np.block(
            [base_function_all_at_t_np * value_y_dot[j], base_function_all_at_t_np * value_x_dot[j]]
        )
    """
    return Q, b


def generate_heading_position_matrix_and_vector(
    base_function: PhysicalSpline,
    time: np.ndarray,
    value_x: np.ndarray,
    value_y: np.ndarray,
    min_dist: float,
    weight_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Matrix Q and the vector b that needs to be added to the optimisation problem,
    to take cartesian positions into account, depends on the helper fit "estimate_locale_heading"

    Args:
        base_function (PhysicalSpline): Spline_model
        time (np.ndarray): time vector must have the same length as the value_x and value_y vector
        value_x (np.ndarray):  x value vector must have the same length as time and value_y vector
        value_y (np.ndarray): y value vector must have the same length as time and value_y vector
        min_dist (float, optional): minimal distance between points for heading estimation. Defaults to 1.
        weight_vec (np.ndarray): weighing each point individuallly must have the same

    Returns:
        tuple[np.array,np.array]: Matrix Q and vector b to be added to the optimization problem
    """

    heading_list = estimate_locale_heading(time, value_x, value_y, min_dist)
    if len(heading_list) == 0:
        return np.zeros((len(base_function.weights) * 2, len(base_function.weights) * 2)), np.zeros(
            len(base_function.weights) * 2
        )
    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, 0)

    Q_one_dimension = generate_1d_weighted_dyadic_product_sum(weight_vec, base_function_all_at_t_np_list)
    b_1 = multiply_each_row_vector_with_weight_and_sum_them_up(
        base_function_all_at_t_np_list, np.sin(heading_list) * weight_vec
    )
    b_2 = multiply_each_row_vector_with_weight_and_sum_them_up(
        base_function_all_at_t_np_list, np.cos(heading_list) * weight_vec
    )

    """
    Q_one_dimension = np.zeros((len(base_function.weights), len(base_function.weights)))
    b_1 = np.zeros(len(base_function.weights))
    b_2 = np.zeros(len(base_function.weights))
    for j in range(len(heading_list)):
        if np.isnan(heading_list[j]):
            continue
        base_function_all_at_t_np = base_function_all_at_t_np_list[j]
        Q_one_dimension += weight_vec[j] * np.outer(np.transpose(base_function_all_at_t_np), base_function_all_at_t_np)
        b_1 += weight_vec[j] * base_function_all_at_t_np * np.sin(heading_list[j])
        b_2 += weight_vec[j] * base_function_all_at_t_np * np.cos(heading_list[j])
    """
    Q = np.block(
        [
            [Q_one_dimension, np.zeros([len(Q_one_dimension), len(Q_one_dimension)])],
            [np.zeros([len(Q_one_dimension), len(Q_one_dimension)]), Q_one_dimension],
        ]
    )
    b = np.block([b_1, b_2])
    return Q, b


def generate_acceleration_heading_matrix_vector(
    base_function: PhysicalSpline,
    time: np.ndarray,
    value_x_dot: np.ndarray,
    value_y_dot: np.ndarray,
    value_x_dotdot: np.ndarray,
    value_y_dotdot: np.ndarray,
    weight_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Matrix Q and the vector b that needs to be added to the optimisation problem,
    Adds Acceleration information. Also needs velocity information because the affect of acceleration on
    yaw rate depends on velocity

    Args:
        base_function (PhysicalSpline): Spline_model
        time (np.ndarray): time vector must have the same length as the value_x and value_y vector
        value_x_dot (np.ndarray): x_dot value vector must have the same length as time and value_y vector
        value_y_dot (np.ndarray): y_dot value vector must have the same length as time and value_y vector
        value_x_dotdot (np.ndarray): x_dotdot value vector must have the same length as time and value_y vector
        value_y_dotdot (np.ndarray): x_dotdot value vector must have the same length as time and value_y vector
        weight_vec (np.ndarray): weighing each point individuallly must have the same

    Returns:
        tuple[np.ndarray, np.ndarray]: Matrix Q and vector b to be added to the optimization problem
    """
    value_x_dot = value_x_dot
    value_y_dot = value_y_dot
    value_x_dotdot = value_x_dotdot
    value_y_dotdot = value_y_dotdot

    base_function_all_at_t_np_list = eval_all_base_all_time(base_function, time, 0)
    base_function_all_at_t_np_list_dot = eval_all_base_all_time(base_function, time, 1)

    vel_vec = np.sqrt(value_x_dot**2 + value_y_dot**2)
    acc_vec = np.where(vel_vec != 0, (value_x_dot * value_x_dotdot + value_y_dot * value_y_dotdot) / vel_vec, 0)
    base_function_vel_acc = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_all_at_t_np_list, acc_vec
    ) + multiply_each_row_of_the_matrix_by_the_nth_vector_entry(base_function_all_at_t_np_list_dot, vel_vec)
    Q_one_dimension = generate_1d_weighted_dyadic_product_sum(weight_vec, base_function_vel_acc)
    base_function_all_at_t_np_weighted = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        base_function_vel_acc, weight_vec
    )
    b_1 = multiply_each_row_vector_with_weight_and_sum_them_up(base_function_all_at_t_np_weighted, value_y_dotdot)
    b_2 = multiply_each_row_vector_with_weight_and_sum_them_up(base_function_all_at_t_np_weighted, value_x_dotdot)
    """
    Q_one_dimension = np.zeros((len(base_function.weights), len(base_function.weights)))
    b_1 = np.zeros(len(base_function.weights))
    b_2 = np.zeros(len(base_function.weights))
    for j in range(len(time)):
        base_functions = base_function_all_at_t_np_list[j]
        base_functions_dot = base_function_all_at_t_np_list_dot[j]

        vel = np.sqrt(value_x_dot[j] ** 2 + value_y_dot[j] ** 2)
        acc = (value_x_dot[j] * value_x_dotdot[j] + value_y_dot[j] * value_y_dotdot[j]) / vel
        c_dyadic_one_dim = acc * base_functions + vel * base_functions_dot
        Q_one_dimension += weight_vec[j] * np.outer(np.transpose(c_dyadic_one_dim), c_dyadic_one_dim)
        b_1 += weight_vec[j] * c_dyadic_one_dim * value_y_dotdot[j]
        b_2 += weight_vec[j] * c_dyadic_one_dim * value_x_dotdot[j]
    """
    Q = np.block(
        [
            [Q_one_dimension, np.zeros([len(Q_one_dimension), len(Q_one_dimension)])],
            [np.zeros([len(Q_one_dimension), len(Q_one_dimension)]), Q_one_dimension],
        ]
    )
    b = np.block([b_1, b_2])
    return Q, b
