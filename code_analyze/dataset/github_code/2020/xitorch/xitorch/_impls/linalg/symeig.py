import warnings
import functools
import torch
from xitorch import LinearOperator
from typing import Union, Optional, Tuple, Sequence
from xitorch._utils.bcast import get_bcasted_dims
from xitorch._utils.tensor import tallqr, to_fortran_order
from xitorch.debug.modes import is_debug_enabled
from xitorch._utils.exceptions import MathWarning

def exacteig(A: LinearOperator, neig: int,
             mode: str, M: Optional[LinearOperator]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition using explicit matrix construction.
    No additional option for this method.

    Warnings
    --------
    * As this method construct the linear operators explicitly, it might requires
      a large memory.
    """
    Amatrix = A.fullmatrix()  # (*BA, q, q)
    if M is None:
        # evals, evecs = torch.linalg.eigh(Amatrix, eigenvectors=True)  # (*BA, q), (*BA, q, q)
        evals, evecs = degen_symeig.apply(Amatrix)  # (*BA, q, q)
        return _take_eigpairs(evals, evecs, neig, mode)
    else:
        Mmatrix = M.fullmatrix()  # (*BM, q, q)

        # M decomposition to make A symmetric
        # it is done this way to make it numerically stable in avoiding
        # complex eigenvalues for (near-)degenerate case
        L = torch.linalg.cholesky(Mmatrix)  # (*BM, q, q)
        Linv = torch.inverse(L)  # (*BM, q, q)
        LinvT = Linv.transpose(-2, -1).conj()  # (*BM, q, q)
        A2 = torch.matmul(Linv, torch.matmul(Amatrix, LinvT))  # (*BAM, q, q)

        # calculate the eigenvalues and eigenvectors
        # (the eigvecs are normalized in M-space)
        # evals, evecs = torch.linalg.eigh(A2, eigenvectors=True)  # (*BAM, q, q)
        evals, evecs = degen_symeig.apply(A2)  # (*BAM, q, q)
        evals, evecs = _take_eigpairs(evals, evecs, neig, mode)  # (*BAM, neig) and (*BAM, q, neig)
        evecs = torch.matmul(LinvT, evecs)
        return evals, evecs

# temporary solution to https://github.com/pytorch/pytorch/issues/47599
class degen_symeig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        eival, eivec = torch.linalg.eigh(A)
        ctx.save_for_backward(eival, eivec)
        return eival, eivec

    @staticmethod
    def backward(ctx, grad_eival, grad_eivec):
        in_debug_mode = is_debug_enabled()

        eival, eivec = ctx.saved_tensors
        min_threshold = torch.finfo(eival.dtype).eps ** 0.6
        eivect = eivec.transpose(-2, -1).conj()

        # remove the degenerate part
        # see https://arxiv.org/pdf/2011.04366.pdf
        if grad_eivec is not None:
            # take the contribution from the eivec
            F = eival.unsqueeze(-2) - eival.unsqueeze(-1)
            idx = torch.abs(F) <= min_threshold
            F[idx] = float("inf")

            # if in debug mode, check the degeneracy requirements
            if in_debug_mode:
                degenerate = torch.any(idx)
                xtg = eivect @ grad_eivec
                diff_xtg = (xtg - xtg.transpose(-2, -1).conj())[idx]
                reqsat = torch.allclose(diff_xtg, torch.zeros_like(diff_xtg))
                # if the requirement is not satisfied, mathematically the derivative
                # should be `nan`, but here we just raise a warning
                if not reqsat:
                    msg = ("Degeneracy appears but the loss function seem to depend "
                           "strongly on the eigenvector. The gradient might be incorrect.\n")
                    msg += "Eigenvalues:\n%s\n" % str(eival)
                    msg += "Degenerate map:\n%s\n" % str(idx)
                    msg += "Requirements (should be all 0s):\n%s" % str(diff_xtg)
                    warnings.warn(MathWarning(msg))

            F = F.pow(-1)
            F = F * torch.matmul(eivect, grad_eivec)
            result = torch.matmul(eivec, torch.matmul(F, eivect))
        else:
            result = torch.zeros_like(eivec)

        # calculate the contribution from the eival
        if grad_eival is not None:
            result += torch.matmul(eivec, grad_eival.unsqueeze(-1) * eivect)

        # symmetrize to reduce numerical instability
        result = (result + result.transpose(-2, -1).conj()) * 0.5
        return result

def davidson(A: LinearOperator, neig: int,
             mode: str,
             M: Optional[LinearOperator] = None,
             max_niter: int = 1000,
             nguess: Optional[int] = None,
             v_init: str = "randn",
             max_addition: Optional[int] = None,
             min_eps: float = 1e-6,
             verbose: bool = False,
             **unused) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Using Davidson method for large sparse matrix eigendecomposition [2]_.

    Arguments
    ---------
    max_niter: int
        Maximum number of iterations
    v_init: str
        Mode of the initial guess (``"randn"``, ``"rand"``, ``"eye"``)
    max_addition: int or None
        Maximum number of new guesses to be added to the collected vectors.
        If None, set to ``neig``.
    min_eps: float
        Minimum residual error to be stopped
    verbose: bool
        Option to be verbose

    References
    ----------
    .. [2] P. Arbenz, "Lecture Notes on Solving Large Scale Eigenvalue Problems"
           http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter12.pdf
    """
    # TODO: optimize for large linear operator and strict min_eps
    # Ideas:
    # (1) use better strategy to get the estimate on eigenvalues
    # (2) use restart strategy

    if nguess is None:
        nguess = neig
    if max_addition is None:
        max_addition = neig

    # get the shape of the transformation
    na = A.shape[-1]
    if M is None:
        bcast_dims = A.shape[:-2]
    else:
        bcast_dims = get_bcasted_dims(A.shape[:-2], M.shape[:-2])
    dtype = A.dtype
    device = A.device

    prev_eigvals = None
    prev_eigvalT = None
    stop_reason = "max_niter"
    shift_is_eigvalT = False
    idx = torch.arange(neig).unsqueeze(-1)  # (neig, 1)

    # set up the initial guess
    V = _set_initial_v(v_init.lower(), dtype, device,
                       bcast_dims, na, nguess,
                       M=M)  # (*BAM, na, nguess)

    best_resid: Union[float, torch.Tensor] = float("inf")
    AV = A.mm(V)
    for i in range(max_niter):
        VT = V.transpose(-2, -1)  # (*BAM,nguess,na)
        # Can be optimized by saving AV from the previous iteration and only
        # operate AV for the new V. This works because the old V has already
        # been orthogonalized, so it will stay the same
        # AV = A.mm(V) # (*BAM,na,nguess)
        T = torch.matmul(VT, AV)  # (*BAM,nguess,nguess)

        # eigvals are sorted from the lowest
        # eval: (*BAM, nguess), evec: (*BAM, nguess, nguess)
        eigvalT, eigvecT = torch.linalg.eigh(T)
        eigvalT, eigvecT = _take_eigpairs(eigvalT, eigvecT, neig, mode)  # (*BAM, neig) and (*BAM, nguess, neig)

        # calculate the eigenvectors of A
        eigvecA = torch.matmul(V, eigvecT)  # (*BAM, na, neig)

        # calculate the residual
        AVs = torch.matmul(AV, eigvecT)  # (*BAM, na, neig)
        LVs = eigvalT.unsqueeze(-2) * eigvecA  # (*BAM, na, neig)
        if M is not None:
            LVs = M.mm(LVs)
        resid = AVs - LVs  # (*BAM, na, neig)

        # print information and check convergence
        max_resid = resid.abs().max()
        if prev_eigvalT is not None:
            deigval = eigvalT - prev_eigvalT
            max_deigval = deigval.abs().max()
            if verbose:
                print("Iter %3d (guess size: %d): resid: %.3e, devals: %.3e" %
                      (i + 1, nguess, max_resid, max_deigval))  # type:ignore

        if max_resid < best_resid:
            best_resid = max_resid
            best_eigvals = eigvalT
            best_eigvecs = eigvecA
        if max_resid < min_eps:
            break
        if AV.shape[-1] == AV.shape[-2]:
            break
        prev_eigvalT = eigvalT

        # apply the preconditioner
        t = -resid  # (*BAM, na, neig)

        # orthogonalize t with the rest of the V
        t = to_fortran_order(t)
        Vnew = torch.cat((V, t), dim=-1)
        if Vnew.shape[-1] > Vnew.shape[-2]:
            Vnew = Vnew[..., :Vnew.shape[-2]]
        nadd = Vnew.shape[-1] - V.shape[-1]
        nguess = nguess + nadd
        if M is not None:
            MV_ = M.mm(Vnew)
            V, R = tallqr(Vnew, MV=MV_)
        else:
            V, R = tallqr(Vnew)
        AVnew = A.mm(V[..., -nadd:])  # (*BAM,na,nadd)
        AVnew = to_fortran_order(AVnew)
        AV = torch.cat((AV, AVnew), dim=-1)

    eigvals = best_eigvals  # (*BAM, neig)
    eigvecs = best_eigvecs  # (*BAM, na, neig)
    return eigvals, eigvecs

def _set_initial_v(vinit_type: str,
                   dtype: torch.dtype, device: torch.device,
                   batch_dims: Sequence,
                   na: int,
                   nguess: int,
                   M: Optional[LinearOperator] = None) -> torch.Tensor:

    torch.manual_seed(12421)
    if vinit_type == "eye":
        nbatch = functools.reduce(lambda x, y: x * y, batch_dims, 1)
        V = torch.eye(na, nguess, dtype=dtype, device=device).unsqueeze(
            0).repeat(nbatch, 1, 1).reshape(*batch_dims, na, nguess)
    elif vinit_type == "randn":
        V = torch.randn((*batch_dims, na, nguess), dtype=dtype, device=device)
    elif vinit_type == "random" or vinit_type == "rand":
        V = torch.rand((*batch_dims, na, nguess), dtype=dtype, device=device)
    else:
        raise ValueError("Unknown v_init type: %s" % vinit_type)

    # orthogonalize V
    if isinstance(M, LinearOperator):
        V, R = tallqr(V, MV=M.mm(V))
    else:
        V, R = tallqr(V)
    return V

def _take_eigpairs(eival, eivec, neig, mode):
    # eival: (*BV, na)
    # eivec: (*BV, na, na)
    if mode == "lowest":
        eival = eival[..., :neig]
        eivec = eivec[..., :neig]
    else:  # uppest
        eival = eival[..., -neig:]
        eivec = eivec[..., -neig:]
    return eival, eivec
