"""This document contains the four methods to approximate the IF
"""
import torch
from src.utils import avg_hvp, hvp_fn, multi_loss_fn, gather_flat_grad, epoch_loss
import numpy as np
import logging

# Identity
def identity(target, model, device, train_loader, regularization_param, eps, it_max, task='zsre', loss_at_epoch=False, break_early=False):
    """Applies the identity matrix as the Hessian.
    Args:
        target: Vector multiplied by inverse hessian (gradient of loss at the point of interest).
    Returns:
        The result of using the identity matrix as the Hessian, resulting in an approximation for H^{-1}(target).
    """
    logging.info(f"Using the identity as the hessian")
    # x = I^{-1} grad_loss = grad_loss
    x = target
    # track loss
    loss = epoch_loss(x, train_loader, model,
                                device, target, regularization_param, task=task)

    return gather_flat_grad(x), loss

# Conjugate Gradient
def conjugate_gradient(x_init, target, model, device, train_loader, regularization_param, eps, it_max, task='zsre', loss_at_epoch=False, break_early=False):
    """Applies conjugate gradient algorithm.
    Args:
        x_init: Initial guess.
        it_max: Maximum number of iterations.
        regularization_param: Constant value for the norm of each iteration.
        target: Vector multiplied by inverse hessian (gradient of loss at the point of interest).
        eps: Tolerance used to detect early termination.
    Returns:
        The result of the conjugate gradient iteration, resulting in an approximation for H^{-1}(target).
    """
    logging.info(f"Max Iterations: {it_max}")
    logging.info(f"Epsilon: {eps}")
    logging.info(f"regularization_param: {regularization_param}")
    loss_list = []
    x = x_init
    i = 0
    residual = [t-h - regularization_param*x0 for t, h,
                x0 in zip(target, avg_hvp(train_loader, model, x,  device, task=task, break_early=break_early), x)]
    direction = residual
    deltanew = sum([torch.dot(torch.flatten(r), torch.flatten(r))
                   for r in residual])
    delta0 = deltanew
    while i < it_max and deltanew > eps ** 2 * delta0:
        alpha = deltanew / sum([torch.dot(torch.flatten(d), torch.flatten(h + regularization_param*d))
                               for d, h in zip(direction, avg_hvp(train_loader, model, direction, device, task=task, break_early=break_early))])
        x = [a + alpha*d for a, d in zip(x, direction)]
        residual = [t-h - regularization_param*x0 for t, h,
                    x0 in zip(target, avg_hvp(train_loader, model, x, device, task=task, break_early=break_early), x)]
        deltaold = deltanew
        deltanew = sum([torch.dot(torch.flatten(r), torch.flatten(r))
                       for r in residual])
        beta = deltanew/deltaold
        direction = [r + beta*d for r, d in zip(residual, direction)]
        i = i+1
        if loss_at_epoch:
            epoch_loss_val = epoch_loss(
                x, train_loader, model, device, target, regularization_param, task=task)
            loss_list.append(epoch_loss_val)
        logging.info(f"Iterations Completed: {i}")
    # track loss
    loss_list.append(epoch_loss(x, train_loader, model,
                                device, target, regularization_param, task=task))

    return gather_flat_grad(x), loss_list

# Stochastic Gradient Descent
def sgd(target, model, device, train_loader,  regularization_param, lr, num_epochs, task='zsre', loss_at_epoch=False, break_early=False):
    """Applies stochastic gradient descent algorithm.
    Args:
        target: Vector multiplied by inverse hessian (gradient of loss at the point of interest).
        lr: Learning rate.
        lambda 1: Constant value for the norm of each iteration.
        num_epochs: Number of epochs.
    Returns:
        The result of the stochastic gradient descent iteration, resulting in an approximation for H^{-1}(target).
    """
    logging.info(f"Number of Epochs: {num_epochs}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"regularization_param: {regularization_param}")
    x = [torch.zeros_like(p) for p in model.parameters()]
    loss_list = []
    for ep in range(num_epochs):
        for j, batch in enumerate(train_loader):
            model.zero_grad()
            train_loss = multi_loss_fn(model, batch, task)
            hvp = hvp_fn(model, train_loss, x)
            x = [xt - (lr*(Hxt + regularization_param*xt - v0))
                 for Hxt, xt, v0 in zip(hvp, x, target)]
            if (break_early == True) & (j == len(train_loader)//2):
                break
        if loss_at_epoch:
            epoch_loss_val = epoch_loss(
                x, train_loader, model, device, target, regularization_param, task=task).item()
            print("\t", ep+1, epoch_loss_val.item())
            loss_list.append(epoch_loss_val)
            if epoch_loss_val > 1e4 or np.isnan(epoch_loss_val):
                logging.info("Loss very large. Quitting training here")
                print("Loss very large. Quitting training here")

                break
            if epoch_loss_val < -1e3:
                logging.info("Loss is unbounded below. Quitting training here")
                print("Loss is unbounded below. Quitting training here")

                break
        logging.info(f"Epochs Completed: {ep}")
    loss_total = epoch_loss(x, train_loader, model,
                            device, target, regularization_param, task=task)
    loss_list.append(loss_total)
    return gather_flat_grad(x), loss_list

# Arnoldi
def arnoldi_iter(start_vector, model, device, train_loader, regularization_param, n_iters, task='zsre', verbose=True, norm_constant=1.0, stop_tol=1e-6):
    """Applies Arnoldi's algorithm.
    Args:
        start_vector: Initial guess.
        n_iters: Number of Arnoldi iterations.
        norm_constant: Constant value for the norm of each projection. In some
            situations (e.g. with a large numbers of parameters) it might be advisable
            to set norm_constant > 1 to avoid dividing projection components by a
            large normalization factor.
        stop_tol: Tolerance used to detect early termination.
    Returns:
        The result of the Arnoldi Iteration, containing the Hessenberg
        matrix A' (n_iters x n_iters-1) approximating A
        on the Krylov subspace K, and the projections onto K. If A is Hermitian
        A' will be tridiagonal (up to numerical errors).
    """

    # Initialization.
    proj = []  # Build the Krylov subspace: w's from the paper
    # Matrix ``A'' in the paper: dot products for Kylov subspace
    appr_mat = torch.zeros((n_iters, n_iters - 1))
    v0_norm = torch.linalg.norm(torch.stack(
        [torch.linalg.norm(v.view(-1)) for v in start_vector]))
    for sv in start_vector:
        sv /= v0_norm
    proj.append(start_vector)
    for n in range(n_iters - 1):
        if verbose:
            if n % 10 == 0:
                print('Starting Arnoldi iteration:', n)
        for i, p in enumerate(proj[n]):
            proj[n][i] = p.to(device)
        vec = [h + regularization_param*p for h,
               p in zip(avg_hvp(train_loader, model, proj[n], device, task=task), proj[n])]
        for i, p in enumerate(proj[n]):
            proj[n][i] = p.to('cpu')
        for i, v in enumerate(vec):
            vec[i] = v.to('cpu')
        for j, proj_vec in enumerate(proj):
            appr_mat[j, n] = sum([torch.dot(torch.flatten(v), torch.flatten(
                pv)) / norm_constant ** 2 for v, pv in zip(vec, proj_vec)])
            for v, pv in zip(vec, proj_vec):
                v -= appr_mat[j, n] * pv
        new_norm = torch.linalg.norm(torch.stack(
            [torch.linalg.norm(v.view(-1)) for v in vec]))

        # Early termination if the Krylov subspace is invariant within the tolerance
        if new_norm < stop_tol:
            appr_mat[n+1, n] = 0
            proj.append(vec)
            break

        appr_mat[n+1, n] = new_norm / norm_constant
        for v in vec:
            v /= appr_mat[n+1, n]

        proj.append(vec)
        if verbose:
            print(f'Finished Arnoldi iteration {n}. Norm = {new_norm}')

    return appr_mat, proj


@torch.no_grad()
def distill(result, top_k: int, verbose=True):
    """Distills the result of an Arnoldi iteration to top_k eigenvalues / vectors.
    Args:
        result: Output of the Arnoldi iteration.
        top_k: How many eigenvalues / vectors to distill.
    Returns:
        An ArnoldiEigens consisting of the distilled eigenvalues and eigenvectors.
        The eigenvectors are conveniently returned in the original basis used
        before applying the Arnoldi iteration and not wrt. the projectors' basis
        selected for the Krylov subspace.
    """
    appr_mat, proj = result
    appr_mat = appr_mat[:-1, :]
    n = appr_mat.shape[0]

    # Make appr_mat symmetric and tridiagonal to avoid complex eigenvals
    for i in range(n):
        for j in range(n):
            if i - j > 1 or j - i > 1:
                appr_mat[i, j] = 0
    # Make symmetric
    appr_mat = (appr_mat + appr_mat.T) / 2
    # Get eigenvalues / vectors for the symmetric matrix appr_mat
    eigvals, eigvecs = torch.linalg.eigh(appr_mat)

    # Sort the eigvals by absolute value.
    idx = torch.argsort(torch.abs(eigvals), descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Note we need to discard the last projector as this is a correction term
    reduced_projections = change_basis_of_projections(
        eigvecs[:, :top_k],  # top-k eigenvectors only
        proj[:-1],  # throw out the last direction
        verbose=verbose)
    # returns list of vectors
    return eigvals[:top_k], reduced_projections


@torch.no_grad()
def change_basis_of_projections(
        matrix,
        proj,
        verbose=False):
    """Changes basis of projections.
    Given a set of projections `proj` and a transformation matrix `matrix`,
    it allows one to obtain new projections from the composition of `matrix` and
    `proj`. For example, the Arnoldi iteration returns a tridiagonal matrix M and
    a set of projections Q; however, to obtain approximate eigenvectors on the
    Krylov subspace, one needs to diagonalize M and use the change of basis matrix
    to its eigenvectors to combine the projections in Q.
    Args:
        matrix: The matrix to use to compose projections.
        proj: The projections.
    Returns:
        The new projections obtained by composing the old ones with matrix.
    """
    if matrix.shape[0] != len(proj):
        raise ValueError('Incompatible composition')
    out = []
    for j in range(matrix.shape[1]):
        if verbose:
            print('Compose projections: j=', j)
        element = [torch.zeros_like(v).to("cpu") for v in proj[0]]
        for i in range(matrix.shape[0]):
            if verbose:
                print(f'Compose projections: i,j={i}, {j}')
            element = [e + matrix[i, j] * p for e, p in zip(element, proj[i])]
        out.append(element)
    return out


def compute_influence_on_loss(gradient, test_gradient, eigvals, eigvecs):
    """Compute the influence of the training point on a test point.
    Args:
        gradient: Gradient of z w.r.t. model params.
        test_gradient: Grad of test function h w.r.t. model params.
    Returns:
        A tensor of <test_gradient, H^{-1} * gradient of train loss>.
    """
    grad = []
    for i, g in enumerate(gradient):
        grad.append(g.clone().detach().to("cpu"))
    # project gradients onto eigvecs
    proj_grad = []
    for ev in eigvecs:
        proj_grad.append(sum([torch.dot(torch.flatten(
            g.clone().detach()), torch.flatten(e)) for g, e in zip(grad, ev)]))
    test_grad = []
    for i, tg in enumerate(test_gradient):
        test_grad.append(tg.clone().detach().to("cpu"))
    proj_test_grad = []
    # project test gradients onto eigvecs
    for ev in eigvecs:
        proj_test_grad.append(sum([torch.dot(torch.flatten(
            g.clone().detach()), torch.flatten(e)) for g, e in zip(test_grad, ev)]))
    # In this space, the Hessian is simply the matrix diagonal given by eigvals
    inverse_hessian_diag = 1 / eigvals
    return torch.dot(gather_flat_grad(proj_test_grad), inverse_hessian_diag * gather_flat_grad(proj_grad))


# SVRG
def variance_reduction(target, model, device, train_loader, regularization_param, lr, num_epochs, task='zsre', loss_at_epoch=False):
    """Applies stochastic variance reduction gradient (svrg) algorithm.
    Args:
        target: Vector multiplied by inverse hessian (gradient of loss of the point of interest).
        regularization_param: Constant value for the norm of each iteration.
        lr: Learning rate.
        num_epochs: Number of epochs.
    Returns:
        The result of the svrg iteration, resulting in an approximation for H^{-1}(target).
    """
    logging.info(f"Number of Epochs: {num_epochs}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"regularization_param: {regularization_param}")

    def batch_gradient_fn(X):
        HX = avg_hvp(train_loader, model, X, device, task=task)
        return [hx + regularization_param * x - v0 for (hx, x, v0) in zip(HX, X, target)]

    def gradient_fn(batch, X):
        model.zero_grad()
        train_loss = multi_loss_fn(model, batch, task)
        HX = hvp_fn(model, train_loss, X)  # \hat H * x
        return [hx + regularization_param * x - v0 for (hx, x, v0) in zip(HX, X, target)]

    X = [torch.zeros_like(v0) for v0 in target]  # x
    loss_list = []

    for ep in range(num_epochs):
        X0 = [torch.clone(x) for x in X]  # checkpoint at the start of an epoch
        batch_grad = batch_gradient_fn(X0)  # batch gradient at the checkpoint
        for j, batch in enumerate(train_loader):
            G = gradient_fn(batch, X)
            G0 = gradient_fn(batch, X0)
            X = [x - lr * (g - g0 + bg)
                 for (x, g, g0, bg) in zip(X, G, G0, batch_grad)]
        if loss_at_epoch:
            epoch_loss_val = epoch_loss(
                X, train_loader, model, device, target, regularization_param, task=task)
            loss_list.append(epoch_loss_val)
            print("Function Loss: ", epoch_loss_val.item())
        logging.info(f"Epochs Completed: {ep}")
    loss_total = epoch_loss(X, train_loader, model,
                            device, target, regularization_param, task=task)
    loss_list.append(loss_total)
    return gather_flat_grad(X), loss_list
