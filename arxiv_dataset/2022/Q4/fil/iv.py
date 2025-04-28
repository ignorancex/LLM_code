from __future__ import annotations
from typing import Union

import jax
from jax import numpy as np
import numpy as onp

from utils import *
from kernels import *


def _matmul_with_Kxy(A: np.ndarray, Kx: Kernel, X: np.ndarray,
                     Y: Union[np.ndarray, None] = None,
                     Y_deriv: bool = False,
                     blk_size=5000) -> np.ndarray:
    """ evaluate A @ Kxy without constructing the latter """
    if Y is None:
        Y = X
    n = X.shape[0]
    def step_fn(t):
        i, prev = t
        A_slice = jax.lax.dynamic_slice(A, (0, i), (A.shape[0], blk_size))
        X_slice = jax.lax.dynamic_slice(X, (i, 0), (blk_size, X.shape[1]))
        K_slice = Kx(X_slice, Y, rhs_deriv=Y_deriv)
        return i+blk_size, prev+A_slice@K_slice
    A_Kxx = 0
    li = n // blk_size * blk_size
    if n >= blk_size:
        R0 = np.zeros((A.shape[0], Y.shape[0]), dtype=A.dtype)
        A_Kxx += jax.lax.while_loop(lambda t: t[0]<li, step_fn, (0, R0))[1]
    if n % blk_size != 0:
        A_Kxx += A[:, li: li+blk_size] @ Kx(X[li: li+blk_size], Y, rhs_deriv=Y_deriv)
    return A_Kxx


class LRLinearMap(object):
    """
    represents A @ B
    """
    def __init__(self, A, B):
        self.A, self.B = (A, B)

    def __call__(self, b: np.ndarray) -> np.ndarray:
        assert len(b.shape) == 2, b.shape
        if self.B is None:
            return self.A @ b
        return self.A @ (self.B @ b)

    def to_onp(self):
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                setattr(self, k, onp.asarray(v))
                del v  # np arrays have a default ref count of 2

    def to_jnp(self):
        for k, v in vars(self).items():
            if isinstance(v, onp.ndarray):
                setattr(self, k, np.asarray(v))

    def __sub__(self, b):
        return LRLinearMap(self.A-b.A, self.B-b.B)

    def _matmul_Kxy(self, Kx: Kernel, X: np.ndarray, Y=None, Y_deriv=False) -> LRLinearMap:
        if Y is None:
            Y = X
        if self.B is None:
            return LRLinearMap(self.A @ Kx(X, Y, rhs_deriv=Y_deriv), None)
        return LRLinearMap(self.A, _matmul_with_Kxy(self.B, Kx, X, Y=Y, Y_deriv=Y_deriv))

    def _matmul_lr(self, b: LRLinearMap) -> LRLinearMap:
        if self.B is None:
            return LRLinearMap(b.T(self.A.T).T, None)
        r = self.B @ b.A if b.B is None else self.B @ b.A @ b.B
        return LRLinearMap(self.A, r)

    def diag(self) -> np.ndarray:
        if self.B is None:
            return np.diag(self.A)
        return np.einsum('ij,ji->i', self.A, self.B)

    def T_(self) -> LRLinearMap:
        if self.B is None:
            return LRLinearMap(self.A.T, None)
        return LRLinearMap(self.B.T, self.A.T)

    T = property(T_)


def add_jitter(A: np.ndarray, jitter: float):
    I = np.eye(A.shape[0])
    evals = np.linalg.eigvalsh(A)
    return A + max(evals[-1]*jitter, -evals[0]+1e-10) * I


def get_nystrom_L(z_nystrom_fn, Z1, Kz, nu, Z2=None, cg=False, sym=False, jitter=1e-12):
    """
    return the evaluation on Z2 of the Nystrom predictor fitted on (Z1m, Z1, f(Z1)=?), which is
        ? |-> K(Z2, Z1m) @ (nu Kmm + Kmn @ Knm)^{-1} @ K(Z1m, Z1) @ ?
    """
    if Z2 is None:
        Z2 = Z1
    Z1m = z_nystrom_fn(Z1)
    Kmn = Kz(Z1m, Z1)
    C = add_jitter(nu * Kz(Z1m, Z1m) + Kmn @ Kmn.T, jitter)
    if not sym:  # old behavior
        B = (cg_solve if cg else np.linalg.solve)(C, Kmn)
        assert not np.isnan(B.sum())
        return LRLinearMap(Kz(Z2, Z1m), B)
    else:
        CL = np.linalg.cholesky(C)  # CL CL.T == C; CL.-T CL.-1 == C^{-1}
        L_sqrt = (cg_solve if cg else np.linalg.solve)(CL, Kmn)
        return LRLinearMap(L_sqrt.T, L_sqrt)

def get_linear_L_sqrt(Kz, Z, nu):
    """
    return: A s.t. A^T A = L
    """
    assert isinstance(Kz, LinearKernel)
    PhiZ = Kz.rf_expand(None, None, Z)
    # for finite-rank kernel (e.g. discrete instruments) there could be numerical issues
    A0 = np.linalg.cholesky(
        add_jitter(PhiZ.T @ PhiZ + nu * np.eye(PhiZ.shape[1]), 1e-20))  # A0 A0.T = ()
    assert not np.isnan(A0.sum())
    return jax.scipy.linalg.solve_triangular(A0, PhiZ.T, lower=True)


class KIVPredictor(object):

    def __init__(self, Z, X, Y, Kz, Kx, lam, nu, z_nystrom=None, z_linear=False, jitter=1e-12):
        n = Z.shape[0]
        vars(self).update(locals())  # save data and args
        self._is_low_rank = z_nystrom is not None or z_linear
        if not self._is_low_rank:
            self._is_low_rank = False
            Kxx = Kx(X, X)
            Kzz = Kz(Z, Z)
            I = np.eye(n)
            L = np.linalg.solve(Kzz + nu*I, Kzz)
            self.eff_prec = LRLinearMap(np.linalg.solve(lam*I + L@Kxx, L), None)
            self.L = LRLinearMap(L, None)
        else:
            """
            Now L is low-rank.  Let A be s.t. A^T A = L, then
            eff_prec = (lambda I + A^T A K_x)^{-1} A^T A 
                     = lam^{-1} (A^T A - A^T (lam I + A Kx A^T)^{-1} A K_x A^T A)
                     =:lam^{-1} A^T tmp A
            """
            if z_linear:  # L = PhiZ (PhiZ^T PhiZ + nu I)^{-1} PhiZ^T
                A = get_linear_L_sqrt(Kz, Z, nu)
                self.L = LRLinearMap(A.T, A)
            else:
                nys_z = z_nystrom(Z)
                Kmn, Kmm = Kz(nys_z, Z), Kz(nys_z, nys_z); Knm = Kmn.T
                Km_tilde = add_jitter(nu * Kmm + Kmn @ Knm, jitter)
                # Let A^T A = L_tilde = Knm Kmtilde^{-1} Kmn
                A0 = np.linalg.cholesky(Km_tilde) # A0 A0.T = Km_tilde; A0^-T A0^-1 = ()^{-1}
                assert not np.isnan(A0.sum())
                A = jax.scipy.linalg.solve_triangular(A0, Kmn, lower=True)
                self.L = get_nystrom_L(z_nystrom, Z, Kz, nu, jitter=jitter, sym=True)
            # calculate eff_prec
            # evals, evecs = np.linalg.eigh(A @ Kx(X, X) @ A.T)
            evals, evecs = np.linalg.eigh(_matmul_with_Kxy(A, Kx, X) @ A.T)
            tmp = evecs @ np.diag(lam / (evals + lam)) @ evecs.T
            assert not np.isnan(tmp.sum())
            self.eff_prec = LRLinearMap(1/lam*A.T, tmp @ A)
        # 
        self.mean_base = self.eff_prec(Y)

    def to_onp(self):
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                setattr(self, k, onp.asarray(v))
                if k not in ['Z', 'X', 'Y', 'lam', 'nu']:
                    del v  # np arrays have a default ref count of 2
        self.eff_prec.to_onp()
        self.L.to_onp()

    def to_jnp(self):
        for k, v in vars(self).items():
            if isinstance(v, onp.ndarray):
                setattr(self, k, np.asarray(v))
        self.eff_prec.to_jnp()
        self.L.to_jnp()

    def _get_cov_subtacted_part(self, x_test, is_deriv=False) -> LRLinearMap:
        assert self._is_low_rank
        return self.eff_prec._matmul_Kxy(
            self.Kx, self.X, Y=x_test, Y_deriv=is_deriv).T._matmul_Kxy(
                self.Kx, self.X, Y=x_test, Y_deriv=is_deriv).T

    def __call__(self, x_test, full_cov=False, evaluate_on_deriv=False):
        """
        return: (pred_mean, pred_cov) if full_cov, else (pred_mean, pred_cov_diag)
        NOTE: pred_mean has shape [N, 1], but pred_cov_diag will have shape [N]
        """
        if not self._is_low_rank:
            Ktx = self.Kx(x_test, self.X, lhs_deriv=evaluate_on_deriv)
            mean = Ktx @ self.mean_base
            if not full_cov:
                Ktt_diag = self.Kx.kdiag(x_test, is_deriv=evaluate_on_deriv)
                cov_ret = Ktt_diag - np.diag(Ktx @ self.eff_prec(Ktx.T))
            else:
                Ktt = self.Kx(
                    x_test, x_test, lhs_deriv=evaluate_on_deriv, rhs_deriv=evaluate_on_deriv)
                cov_ret = Ktt - Ktx @ self.eff_prec(Ktx.T)
            return mean, cov_ret
        #
        mean_T = _matmul_with_Kxy(
            self.mean_base.T, self.Kx, self.X, Y=x_test, Y_deriv=evaluate_on_deriv)
        csp = self._get_cov_subtacted_part(x_test, is_deriv=evaluate_on_deriv)
        if full_cov:
            return mean_T.T, (csp, 'handle it manually')
        return mean_T.T, self.Kx.kdiag(x_test, is_deriv=evaluate_on_deriv) - csp.diag()
 
    def log_qlh(self):
        """
        return log \int Pi(df) exp(-(f(X)-Y) lam L^{-1} (f(X)-Y)).
        """
        quad_term = (self.Y * self.eff_prec(self.Y)).sum()
        # calculate logdet := log |lam^{-1} Kx L + I| for L = L.A @ L.B
        if not isinstance(self.Kx, LinearKernel):
            Kxs = np.linalg.cholesky(add_jitter(self.Kx(self.X, self.X), 1e-7))
        else:
            Kxs = self.Kx.rf_expand(None, None, self.X)
        # lkx = self.L._matmul_Kxy(self.Kx, self.X); lkx_ = lkx.B @ lkx.A
        lkx_s = self.L.B @ Kxs
        lkx_ = add_jitter(lkx_s @ lkx_s.T, 1e-10)  # for occasional failure in small samples
        evals = np.linalg.eigvalsh(lkx_)
        logdet = np.sum(np.log((evals + self.lam) / self.lam))
        return -1/2 * (logdet + quad_term)

    
def kiv(Z, X, Y, Kz, Kx, lam, nu, return_log_qlh=False, z_nystrom=None, z_linear=False,
        jitter=1e-9):
    pred = KIVPredictor(
        Z, X, Y, Kz, Kx, lam, nu, z_nystrom=z_nystrom, z_linear=z_linear, jitter=jitter)
    if not return_log_qlh:
        return pred
    return pred, pred.log_qlh()


def kiv_stage1_criteria(zx_tuples, nu, Kz, Kx, z_nystrom=None, z_linear=False,
                        cg=False, jitter=1e-9):
    (Z1, X1), (Z2, X2) = zx_tuples
    assert not z_linear, NotImplementedError()
    if z_nystrom is None:
        Kz11, Kz12, Kx21, Kx11 = (Kz(Z1, Z1), Kz(Z1, Z2), Kx(X2, X1), Kx(X1, X1))
        L = np.linalg.solve(Kz11 + nu*np.eye(Z1.shape[0]), Kz12)
        return Kx.kdiag(X2).mean() + np.diag(-2 * Kx21 @ L + L.T @ Kx11 @ L).mean()
    else:
        L_T = get_nystrom_L(z_nystrom, Z1, Kz, nu, Z2, cg=cg, jitter=jitter)
        ret = Kx.kdiag(X2).mean() +\
            -2 * (L_T._matmul_Kxy(Kx, X1, X2).diag()).mean() + \
            L_T._matmul_Kxy(Kx, X1)._matmul_lr(L_T.T).diag().mean()
        return ret


def kiv_hps_selection(
    dtrain, dheldout, Kz, Kx, nu_space, lam_space=None, s2_criterion='orig',
    return_all_stats=False, z_nystrom=None, z_linear=False, jitter=1e-9):
    """
    Hyperparameter selection for KIV. Selection of nu follows Algorithm 2 in the KIV paper.
    For lambda, we implement
    - 'orig': as in the KIV paper.  It seems to be a typo: the KIV stage 2 objective is 
      E (y-Hmu(z))^2 = E(y-E(h(x)|z))^2, but their algorithm minimizes E(y-h(x))^2.
    - 'proj': corrects the above issue.
    - 'nll': negative test quasi-loglh
    """
    if len(nu_space) > 1:
        s1_stats = onp.array(
            [kiv_stage1_criteria(
                (dtrain[:2], dheldout[:2]), nu, Kz, Kx, z_nystrom=z_nystrom,
                z_linear=z_linear, jitter=jitter)
             for nu in nu_space])
        nu = nu_space[onp.nanargmin(s1_stats)]
    else:
        nu = nu_space[0]
    
    if lam_space is None:
        return ((nu, s1_stats) if return_all_stats else nu)
    
    (Z1, X1, Y1), (Z2, X2, Y2) = dtrain, dheldout
    
    if z_nystrom is not None:
        L2 = get_nystrom_L(z_nystrom, Z2, Kz, nu, jitter=jitter)
    elif z_linear:
        L2_sqrt = get_linear_L_sqrt(Kz, Z2, nu)
        L2 = LRLinearMap(L2_sqrt.T, L2_sqrt)
    else:
        K2 = Kz(Z2, Z2)
        L2 = np.linalg.solve(K2 + nu*np.eye(K2.shape[0]), K2).T
        L2 = LRLinearMap(L2, None)

    proj_stats, orig_stats, nll_stats = [], [], []
    for lam in lam_space:
        pred_fn = kiv(Z1, X1, Y1, Kz, Kx, lam, nu, z_nystrom=z_nystrom, z_linear=z_linear,
                      jitter=jitter)
        r"""
        proj: computes
            (\hat E \hat f)(z2) = Sz2 (Cz^{-1} Czx) mu \approx (empirical ver.) =: pf_cexp
        Note that we can use either dtrain or dheldout to estimate the condexp operator,
        as this is validation.  We use dheldout which seems closer to the typical sample 
        splitting regime.  Then we need to calculate L2 S_{x2} mu = L2 @ pred_fn(X2) where
        L2 = S_{z2} \invEmpCz2 n2^{-1}S_{z2} = K_{z2}(K_{z2}+nu I)^{-1}. 
        """
        pfx2_mean, _ = pred_fn(X2)
        pf_cexp = L2(pfx2_mean)
        proj_stats.append(((pf_cexp-Y2)**2).mean())
        """ orig, from aux/KIV2_loss.m of the KIV codebase """
        orig_stats.append(((pfx2_mean-Y2)**2).mean())
        pred_fn.to_onp()  # GC workaround

    proj_stats, orig_stats, nll_stats = map(onp.array, (proj_stats, orig_stats, nll_stats))
    s2_stats = locals()[s2_criterion+'_stats']
    if onp.any(onp.isnan(s2_stats)):
        print('warning: nan encountered in kiv_hps_selection, crit=',
              s2_criterion, lam_space, s2_stats)
    lam = lam_space[onp.nanargmin(s2_stats)]
    
    if return_all_stats:
        return nu, lam, (proj_stats, orig_stats, nll_stats)
    else:
        return nu, lam


def cg_solve(A, b):
    return jax.scipy.sparse.linalg.cg(A, b)[0]


def krr(x, y, k, lam, cg=False, nystrom=None, linear=False, jitter=1e-9):
    solve = np.linalg.solve if not cg else cg_solve
    if nystrom is None and not linear:
        kx = k(x, x)
        mean_base = solve(kx + lam*np.eye(kx.shape[0]), y)
        def predict(xtest):
            return k(xtest, x) @ mean_base
    elif not linear:
        x_nys = nystrom(x)
        L = get_nystrom_L(lambda _: x_nys, x, k, lam, cg=cg, jitter=jitter)
        def predict(xtest):
            return k(xtest, x_nys) @ (L.B @ y)
    else:
        # \hat f(X_te) = PhiZ_te (PhiZ^T PhiZ + nu I)^{-1} PhiZ^T Y =: PhiZ_te @ mean_base
        Phi_tr = k.rf_expand(None, None, x)
        mean_base = solve(Phi_tr.T @ Phi_tr + lam * np.eye(Phi_tr.shape[1]), Phi_tr.T @ y)
        def predict(xtest):
            return k.rf_expand(None, None, xtest) @ mean_base
    return predict

