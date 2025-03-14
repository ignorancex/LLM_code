# file mostly from SciPy:
# https://github.com/scipy/scipy/blob/914523af3bc03fe7bf61f621363fca27e97ca1d6/scipy/optimize/nonlin.py#L221
# and converted to PyTorch for GPU efficiency

from typing import Dict, Any, Optional, Union, Tuple
import warnings
import torch
import functools
from xitorch._impls.optimize.root._jacobian import NewtonJacobian, BroydenFirst, \
    BroydenSecond, LinearMixing, Jacobian
from xitorch._utils.exceptions import ConvergenceWarning

__all__ = ["newton", "broyden1", "broyden2", "linearmixing"]

def _nonlin_solver(fcn, x0, params,
                   # jacobian
                   jacobian: Jacobian,
                   # stopping criteria
                   maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                   # algorithm parameters
                   line_search=True,
                   # misc parameters
                   verbose=False,
                   custom_terminator=None,
                   **unused):
    """
    Keyword arguments
    -----------------
    maxiter: int or None
        Maximum number of iterations, or inf if it is set to None.
    f_tol: float or None
        The absolute tolerance of the norm of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the norm of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    line_search: bool or str
        Options to perform line search. If ``True``, it is set to ``"armijo"``.
    verbose: bool
        Options for verbosity
    """

    if maxiter is None:
        maxiter = 100 * (torch.numel(x0) + 1)
    if line_search is True:
        line_search = "armijo"
    elif line_search is False:
        line_search = None

    # solving complex rootfinder by concatenating real and imaginary part,
    # making the variable twice as long
    x_is_complex = torch.is_complex(x0)

    def _ravel_complex(x: torch.Tensor) -> torch.Tensor:
        # represents complex x as a long real vector
        return torch.cat((x.real, x.imag), dim=0).reshape(-1)

    def _pack_complex(x: torch.Tensor) -> torch.Tensor:
        # pack a long real vector into a complex vector with the shape accepted by fcn
        n = len(x) // 2
        xreal, ximag = x[:n], x[n:]
        x = xreal + 1j * ximag
        return x.reshape(xshape)

    _ravel = _ravel_complex if x_is_complex else (lambda x: x.reshape(-1))
    _pack = _pack_complex if x_is_complex else (lambda x: x.reshape(xshape))

    # shorthand for the function
    xshape = x0.shape
    func = lambda x: _ravel(fcn(_pack(x), *params))
    x = _ravel(x0)

    y = func(x)
    y_norm = y.norm()
    stop_cond = custom_terminator if custom_terminator is not None \
        else TerminationCondition(f_tol, f_rtol, y_norm, x_tol, x_rtol)
    if (y_norm == 0):
        return x.reshape(xshape)

    # set up the jacobian
    jacobian.setup(x, y, func)

    # solver tolerance
    gamma = 0.9
    eta_max = 0.9999
    eta_threshold = 0.1
    eta = 1e-3

    converge = False
    best_ynorm = y_norm
    best_x = x
    best_dxnorm = x.norm()
    best_iter = 0
    for i in range(maxiter):
        tol = min(eta, eta * y_norm)
        dx = -jacobian.solve(y, tol=tol)

        dx_norm = dx.norm()
        if dx_norm == 0:
            raise ValueError("Jacobian inversion yielded zero vector. "
                             "This indicates a bug in the Jacobian "
                             "approximation.")

        if line_search:
            s, xnew, ynew, y_norm_new = _nonline_line_search(func, x, y, dx,
                                                             search_type=line_search)
        else:
            s = 1.0
            xnew = x + dx
            ynew = func(xnew)
            y_norm_new = ynew.norm()

        # save the best results
        if y_norm_new < best_ynorm:
            best_x = xnew
            best_dxnorm = dx_norm
            best_ynorm = y_norm_new
            best_iter = i + 1

        jacobian.update(xnew.clone(), ynew)

        # print out dx and df
        to_stop = stop_cond.check(xnew, ynew, dx)
        if verbose:
            if i < 10 or i % 10 == 0 or to_stop:
                print("%6d: |dx|=%.3e, |f|=%.3e" % (i, dx_norm, y_norm))
        if to_stop:
            converge = True
            break

        # adjust forcing parameters for inexact solve
        eta_A = float(gamma * (y_norm_new / y_norm)**2)
        gamma_eta2 = gamma * eta * eta
        if gamma_eta2 < eta_threshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma_eta2))

        y_norm = y_norm_new
        x = xnew
        y = ynew
    if not converge:
        msg = ("The rootfinder does not converge after %d iterations. "
               "Best |dx|=%.3e, |f|=%.3e at iter %d") % (maxiter, best_dxnorm, best_ynorm, best_iter)
        warnings.warn(ConvergenceWarning(msg))
        x = best_x
    return _pack(x)

def newton(fcn, x0, params=(), *,
           solver_method: str = "exactsolve",
           solver_kwargs: Optional[Dict[str, Any]] = None,
           **kwargs):
    """
    Solve the root finder using the Newton method. In root finding task, Newton's method goes as follows:

    .. math::

        x_{n+1} = x_n - J^{-1}(x_n) f(x_n)

    where :math:`J` is the Jacobian of the function :math:`f`.

    Keyword arguments
    -----------------
    solver_method: str
        The method to solve the linear equation (i.e., the method in computing the inverse of Jacobian above).
        The list of available methods can be found in :class:`xitorch.linalg.solve`.
    solver_kwargs: dict or None
        The keyword arguments for the solver method. The list of available keyword arguments
        can be found in :class:`xitorch.linalg.solve`.
    """
    jacobian = NewtonJacobian(solver_method=solver_method, solver_kwargs=solver_kwargs)
    return _nonlin_solver(fcn, x0, params, jacobian=jacobian, **kwargs)

def broyden1(fcn, x0, params=(), *,
             alpha: Optional[float] = None,
             uv0: Optional[Union[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
             max_rank: Optional[int] = None,
             **kwargs):
    """
    Solve the root finder or linear equation using the first Broyden method [1]_.
    It can be used to solve minimization by finding the root of the
    function's gradient.

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional systems of nonlinear equations".
           Mathematisch Instituut, Universiteit Leiden, The Netherlands (2003).
           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Keyword arguments
    -----------------
    alpha: float or None
        The initial guess of inverse Jacobian is ``- alpha * I + u v^T``.
    uv0: tuple of tensors or str or None
        The initial guess of inverse Jacobian is ``- alpha * I + u v^T``.
        If ``"svd"``, then it uses 1-rank svd to obtain ``u`` and ``v``.
        If None, then ``u`` and ``v`` are zeros.
    max_rank: int or None
        The maximum rank of inverse Jacobian approximation. If ``None``, it
        is ``inf``.
    """
    jacobian = BroydenFirst(alpha=alpha, uv0=uv0, max_rank=max_rank)
    return _nonlin_solver(fcn, x0, params, jacobian=jacobian, **kwargs)

@functools.wraps(_nonlin_solver, assigned=('__annotations__',))  # takes only the signature
def broyden2(fcn, x0, params=(), *,
             alpha: Optional[float] = None,
             uv0: Optional[Union[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
             max_rank: Optional[int] = None,
             **kwargs):
    """
    Solve the root finder or linear equation using the second Broyden method [2]_.
    It can be used to solve minimization by finding the root of the
    function's gradient.

    References
    ----------
    .. [2] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional systems of nonlinear equations".
           Mathematisch Instituut, Universiteit Leiden, The Netherlands (2003).
           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Keyword arguments
    -----------------
    alpha: float or None
        The initial guess of inverse Jacobian is ``- alpha * I + u v^T``.
    uv0: tuple of tensors or str or None
        The initial guess of inverse Jacobian is ``- alpha * I + u v^T``.
        If ``"svd"``, then it uses 1-rank svd to obtain ``u`` and ``v``.
        If None, then ``u`` and ``v`` are zeros.
    max_rank: int or None
        The maximum rank of inverse Jacobian approximation. If ``None``, it
        is ``inf``.
    """
    jacobian = BroydenSecond(alpha=alpha, uv0=uv0, max_rank=max_rank)
    return _nonlin_solver(fcn, x0, params, jacobian=jacobian, **kwargs)

def linearmixing(fcn, x0, params=(), *,
                 # jacobian parameters
                 alpha: Optional[float] = None,
                 # solver parameters
                 **kwargs):
    """
    Solve the root finding problem by approximating the inverse of Jacobian
    to be a constant scalar.

    Keyword arguments
    -----------------
    alpha: float or None
        The initial guess of inverse Jacobian is ``-alpha * I``.
    """
    jacobian = LinearMixing(alpha=alpha)
    return _nonlin_solver(fcn, x0, params, jacobian=jacobian, **kwargs)


# set the docstring of the functions
newton.__doc__ += _nonlin_solver.__doc__  # type: ignore
broyden1.__doc__ += _nonlin_solver.__doc__  # type: ignore
broyden2.__doc__ += _nonlin_solver.__doc__  # type: ignore
linearmixing.__doc__ += _nonlin_solver.__doc__  # type: ignore

def _norm_sq(v):
    # if not torch.isfinite(v).all():
    #     return torch.tensor(float("inf"), dtype=v.dtype, device=v.device)
    v1 = v.reshape(-1)
    y = torch.dot(v1, v1)
    return y

def _nonline_line_search(func, x, y, dx, search_type="armijo", rdiff=1e-8, smin=1e-2):
    tmp_s = [0]
    tmp_y = [y]
    tmp_phi = [y.norm()**2]
    s_norm = x.norm() / dx.norm()

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]
        xt = x + s * dx
        v = func(xt)
        p = _norm_sq(v)
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_y[0] = v
        return p

    def derphi(s):
        ds = (torch.abs(s) + s_norm + 1) * rdiff
        return (phi(s + ds, store=False) - phi(s)) / ds

    if search_type == 'armijo':
        s, phi1 = _scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0],
                                        amin=smin)

    if s is None:
        # No suitable step length found. Take the full Newton step,
        # and hope for the best.
        s = 1.0

    x = x + s * dx
    if s == tmp_s[0]:
        y = tmp_y[0]
    else:
        y = func(x)
    y_norm = y.norm()

    return s, x, y, y_norm

def _scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0, max_niter=20):
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1 * alpha1 * derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    niter = 0
    while alpha1 > amin and niter < max_niter:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1 * alpha2 * derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
        niter += 1

    # Failed to find a suitable step length
    if niter == max_niter:
        return alpha2, phi_a2
    return None, phi_a1

class TerminationCondition(object):
    def __init__(self, f_tol, f_rtol, f0_norm, x_tol, x_rtol):
        if f_tol is None:
            f_tol = 1e-6
        if f_rtol is None:
            f_rtol = float("inf")
        if x_tol is None:
            x_tol = 1e-6
        if x_rtol is None:
            x_rtol = float("inf")
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f0_norm = f0_norm

    def check(self, x: torch.Tensor, y: torch.Tensor, dx: torch.Tensor) -> bool:
        xnorm = x.norm()
        ynorm = y.norm()
        dxnorm = dx.norm()
        return (dxnorm < self.x_tol) and (dxnorm < self.x_rtol * xnorm) and \
            (ynorm < self.f_tol) and (ynorm < self.f_rtol * self.f0_norm)
