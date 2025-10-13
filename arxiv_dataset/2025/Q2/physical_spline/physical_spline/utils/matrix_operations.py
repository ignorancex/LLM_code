import numpy as np


def multiply_each_row_vector_with_weight_and_sum_them_up(
    matrix_with_vector_in_each_row: np.ndarray, weight_for_each_row: np.ndarray | None = None
) -> np.ndarray:
    """multiplies each row vectors of the matrix with a weight from the weight vector
    Afterwards sums all vectors together

    Args:
        matrix_with_vector_in_each_row (np.ndarray): numpy array
        weight_for_each_row (np.ndarray | None, optional): vector that has the length
        of the matrix columns. Defaults to None.

    Returns:
        np.ndarray: sumed up vector
    """
    if weight_for_each_row is None:
        return np.einsum("ji -> i", matrix_with_vector_in_each_row)
    return np.einsum("ji,j->i", matrix_with_vector_in_each_row, weight_for_each_row)


def multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
    matrix_with_vector_in_each_row: np.ndarray, weight_for_each_row: np.ndarray
) -> np.ndarray:
    """multiplies each row vector of the matrix with a weight

    Args:
        matrix_with_vector_in_each_row (np.ndarray): numpy array
        weight_for_each_row (np.ndarray | None, optional): vector that has the length
        of the matrix columns. Defaults to None.

    Returns:
        np.ndarray: matrix with multiplied row vectors
    """
    return np.einsum("ij, i -> ij", matrix_with_vector_in_each_row, weight_for_each_row)


def sum_up_dyadic_product_of_row_vectors(matrix_with_vector_in_each_row: np.ndarray) -> np.ndarray:
    """builds the dyadic product of the row vectors and sums the resulting matrices together

    Args:
        base_function_all_at_t_np_weighted_sqrt (np.ndarray): _description_

    Returns:
        np.ndarray: sum of the dyadic product of row vectors
    """
    # return np.einsum("ki,kj -> ij", matrix_with_vector_in_each_row, matrix_with_vector_in_each_row)
    return np.matmul(np.transpose(matrix_with_vector_in_each_row), matrix_with_vector_in_each_row)


def generate_1d_weighted_dyadic_product_sum(
    weight_vec: np.ndarray, matrix_with_vector_in_each_row: np.ndarray
) -> np.ndarray:
    """builds the dyadic product of the row vectors and weights the resulting matrices
    and sums the weight matrices together

    Args:
        weight_vec (np.ndarray): weight for each dyadic product
        matrix_with_vector_in_each_row (np.ndarray): _description_

    Returns:
        np.ndarray: sumed up result
    """
    base_function_all_at_t_np_weighted_sqrt = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        matrix_with_vector_in_each_row, np.sqrt(weight_vec)
    )
    return sum_up_dyadic_product_of_row_vectors(base_function_all_at_t_np_weighted_sqrt)


def generate_2d_weighted_dyadic_product_and_sum_them_up(
    weight_vec_x: np.ndarray, weight_vec_y: np.ndarray, matrix_with_vector_in_each_row: np.ndarray
) -> np.ndarray:
    """Similar to 1D but weights each dimension seperatly stacks them as block matrix and then
    sums it up

    Args:
        weight_vec_x (np.ndarray): weight
        weight_vec_y (np.ndarray): _description_
        matrix_with_vector_in_each_row (np.ndarray): _description_

    Returns:
        np.ndarray: sumed up result
    """
    base_function_all_weighted_dim1 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        matrix_with_vector_in_each_row, weight_vec_x
    )
    base_function_all_weighted_dim2 = multiply_each_row_of_the_matrix_by_the_nth_vector_entry(
        matrix_with_vector_in_each_row, weight_vec_y
    )
    return sum_up_dyadic_product_of_row_vectors(
        np.block([base_function_all_weighted_dim1, base_function_all_weighted_dim2])
    )
