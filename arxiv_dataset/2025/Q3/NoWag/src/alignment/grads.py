import torch

# hardcoded grads for the hessian


def grad_quadratic(w_hat, w, hessian):
    """gradient of tr((w-w_hat)H(w-w_hat)^T) with respect to w_hat"""
    return 2 * (w_hat - w) @ hessian


def grad_quadratic_low_rank(A, B, w, hessian):
    """gradient of tr((w-AB^T)H(w-AB^T)^T) with respect to A and B"""

    diff = w - A @ B
    A_grad = -2 * diff @ hessian @ B.t()
    # print((2* A.t() @ A @ B @ H).shape)
    B_grad = -2 * A.t() @ diff @ hessian

    return A_grad, B_grad
