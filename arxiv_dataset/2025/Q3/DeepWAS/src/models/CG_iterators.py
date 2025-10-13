import numpy as np
import torch
from torch.autograd import Function
from torch.func import vjp
from torch.linalg import solve_triangular as sotri

from cola.linalg.inverse.torch_cg import cg
from cola.ops.operators import Dense

from .trainer import SNPTrainerRFR


class SLQ(Function):
    @staticmethod
    def forward(ctx, A, *args, **kwargs):
        output = slq_fwd_grad(A, *args, **kwargs)
        ctx.save_for_backward(A)
        ctx.args = args
        ctx.kwargs = kwargs
        return output

    @staticmethod
    def backward(ctx, *grads):
        (A,) = ctx.saved_tensors
        output = slq_bwd_grad(A, grads, *ctx.args, **ctx.kwargs)
        return output


slq_grad_only_torch = SLQ.apply


def slq_fwd_grad(A, vtol, P, max_iters, tol):
    estimate = torch.randn(A.shape[1], 1, dtype=A.dtype, device=A.device)
    return torch.mean(estimate)


def slq_bwd_grad(A, grads, *args, **kwargs):
    vtol, P, max_iters, tol = args
    num_samples = round(1 / vtol**2.0)

    probes = torch.randn(A.shape[1], num_samples, dtype=A.dtype, device=A.device)
    x0 = torch.zeros_like(probes)
    with torch.no_grad():
        probes_solves, *_ = cg(A, probes, x0=x0, P=P, tol=tol, max_iters=max_iters)
    coef = 1.0 / probes.shape[-1]
    g = grads[0]
    d_solves = coef * g * probes_solves

    def fun(theta):
        return theta @ probes

    dA = vjp_derivs(fun=fun, primals=A, duals=d_solves)
    out = dA + tuple([None] * len(args))
    return out


def vjp_derivs(fun, primals, duals):
    _, vjpfun = vjp(fun, primals)
    output = vjpfun(duals)
    return output


class WASP(Function):
    @staticmethod
    def forward(ctx, A, rhs, P, tol, max_iters, vtol):
        out, logdet, quad, A, all_rhs, all_soln = logdet_quad_fwd(A, rhs, P, tol, max_iters, vtol)
        ctx.save_for_backward(A, all_rhs, all_soln)
        return out, logdet, quad

    @staticmethod
    def backward(ctx, *grads):
        A, all_rhs, all_soln = ctx.saved_tensors
        out = logdet_quad_bwd(grads, A, all_rhs, all_soln)
        return out


wasp_ldq = WASP.apply


def logdet_quad_fwd(A, rhs, P, tol, max_iters, vtol):
    if vtol > 0.0:
        num_samples = round(1 / vtol**2.0)
        probes = torch.randn(A.shape[1], num_samples, dtype=A.dtype, device=A.device)
        coef = 1.0 / probes.shape[-1]
    else:
        probes = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
        coef = 1.0
    all_rhs = torch.concatenate((rhs, probes), dim=-1)
    logdet = 0.0
    AOp, POp = Dense(A), Dense(P)
    all_soln, *_ = cg(AOp, all_rhs, x0=torch.zeros_like(all_rhs), P=POp, tol=tol, max_iters=max_iters)
    all_rhs = torch.concatenate((all_soln[:, [0]], coef * probes), dim=-1)
    quad = torch.sum(rhs * all_soln[:, [0]])
    out = logdet - quad
    return out, logdet, quad, A, all_rhs, all_soln


def logdet_quad_bwd(grads, A, all_rhs, all_soln):
    def fun(theta):
        return theta @ all_soln

    dA = vjp_derivs(fun=fun, primals=A, duals=grads[0] * all_rhs)
    out = dA + tuple([None] * 5)
    return out


class SNPTrainerCG(SNPTrainerRFR):
    def forward(self, betas, R, log_F_N, D, ld_window, sigma):
        n_snp = len(log_F_N)
        tol, max_iters, vtol = self.tol, self.max_iters, self.vtol
        diag = torch.exp(log_F_N)
        Id = torch.eye(ld_window.sum(), dtype=R.dtype, device=R.device)
        A = R[ld_window] @ (diag[:, None] * R[..., ld_window]) + sigma * R[..., ld_window][ld_window] + self.eps * Id
        P = self.compute_precond(R, log_F_N, sigma, ld_window, A)

        with torch.no_grad():
            rhs = (betas * torch.sqrt(self.N))[ld_window][:, None]
            soln, *_ = cg(A, rhs, x0=torch.zeros_like(rhs), P=P, tol=tol, max_iters=max_iters)
            soln, rhs = soln[:, 0], rhs[:, 0]
            rel_res = torch.linalg.norm(A @ soln - rhs) / torch.linalg.norm(rhs)
            rv = R[:, ld_window] @ soln
        err = rv @ (diag * rv) + sigma * rv[ld_window] @ soln + self.eps * soln @ soln
        log_det = slq_grad_only_torch(A, vtol, P, max_iters, tol)
        log_det = log_det - n_snp * torch.log(self.N)
        loss = log_det + (2 * err.detach() - err) + n_snp * np.log(2 * np.pi)
        loss = (0.5 / n_snp) * loss.float()

        with torch.no_grad():
            log_F = log_F_N - torch.log(self.N) + torch.log(self.M)
            F = torch.exp(log_F)

        self.print_info(R, betas, ld_window, diag, log_det, err, sigma, rel_res, n_snp)
        info = {
            "loss": loss,
            "err": err / n_snp,
            "logdet": log_det / n_snp,
            "mean_log_F": log_F.mean().item(),
            "mean_F": F.mean().item(),
            "solve_RpFinv": rel_res,
        }
        return loss, info

    def compute_precond(self, R, log_F, sigma, ld_window, A):
        with torch.no_grad():
            # Lam = torch.ones(R.shape[0], dtype=R.dtype, device=R.device)
            # Q = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
            Lam, Q = torch.linalg.eigh(R)
            mu = torch.exp(log_F).mean()
            diag = mu * Lam**2.0 + sigma * Lam + self.eps
            P_RmR = Q[ld_window] @ torch.diag(1 / diag) @ Q[ld_window].T
        return P_RmR


class WASPBench(SNPTrainerCG):
    def forward(self, betas, R, log_F_N, D, ld, sigma, P, lam, Rinv, L, W):
        # lam, U = torch.linalg.eigh(R[ld][..., ld])
        # mask = lam > 1e-4
        # Rinv = U[:, mask] @ ((1 / lam[mask, None]) * U[:, mask].T)
        # L = R[..., ld] @ Rinv
        # W = L @ R[ld]

        F12 = torch.exp(0.5 * log_F_N)
        B = (1 / sigma) * (F12[:, None] * W) * F12[None, :]
        A = torch.eye(B.shape[0], dtype=B.dtype, device=B.device) + B

        betas_scaled = betas[ld] * torch.sqrt(self.N)
        rhs = (1 / sigma) * (F12[:, None] * L @ betas_scaled)

        loss, log_det, quad = wasp_ldq(A, rhs[:, None], P, self.tol, self.max_iters, self.vtol)

        log_det = log_det + torch.sum(torch.log(sigma * lam))
        quad = (1 / sigma) * betas_scaled.T @ (Rinv @ betas_scaled) - quad

        loss = loss + torch.sum(torch.log(sigma * lam))
        loss = loss + (1 / sigma) * betas_scaled.T @ (Rinv @ betas_scaled)

        n_snp = len(log_F_N)
        loss = loss - n_snp * torch.log(self.N) + n_snp * np.log(2 * np.pi)
        loss = (0.5 / n_snp) * loss

        with torch.no_grad():
            log_F = log_F_N - torch.log(self.N) + torch.log(self.M)
            F = torch.exp(log_F)

        rel_res = 1e-6
        F_N = torch.exp(log_F_N)
        self.print_info(R, betas, ld, F_N, log_det, quad, sigma, rel_res, n_snp)
        info = {
            "loss": loss,
            "err": quad / n_snp,
            "logdet": log_det / n_snp,
            "mean_log_F": log_F.mean().item(),
            "mean_F": F.mean().item(),
            "solve_RpFinv": rel_res,
        }
        return loss, info


class WASPPre(SNPTrainerCG):
    def forward(self, betas, R, log_F_N, D, ld, sigma):
        Ri = R[ld][..., ld]
        Ri = Ri + 1e-4 * torch.eye(Ri.shape[0], dtype=Ri.dtype, device=Ri.device)
        L = torch.linalg.cholesky(Ri)
        RinvRt = sotri(L.T, sotri(L, R[ld], upper=False), upper=True)

        Finv = torch.exp(-log_F_N)
        soln = sotri(L.T, sotri(L, betas[ld][:, None], upper=False), upper=True)[:, 0]
        Up = R[..., ld] @ RinvRt
        A = sigma * (torch.diag(sigma * Finv) + Up)
        rhs = RinvRt.T @ betas[ld] * torch.sqrt(self.N)

        P = self.compute_precond(Up, log_F_N, sigma, ld, A)
        loss, log_det, err = wasp_ldq(A, rhs[:, None], P, self.tol, self.max_iters, self.vtol)
        loss = loss + log_det

        loss = loss + torch.logdet(Ri) + log_F_N.sum()
        loss = loss + (Ri.shape[0] - 2 * R.shape[0]) * torch.log(sigma)
        loss = loss + (1 / sigma) * betas[ld].T @ soln

        n_snp = len(log_F_N)
        loss = loss - n_snp * torch.log(self.N) + n_snp * np.log(2 * np.pi)
        loss = (0.5 / n_snp) * loss

        with torch.no_grad():
            log_F = log_F_N - torch.log(self.N) + torch.log(self.M)
            F = torch.exp(log_F)

        rel_res = 1e-6
        F_N = torch.exp(log_F_N)
        self.print_info(R, betas, ld, F_N, log_det, err, sigma, rel_res, n_snp)
        info = {
            "loss": loss,
            "err": err / n_snp,
            "logdet": log_det / n_snp,
            "mean_log_F": log_F.mean().item(),
            "mean_F": F.mean().item(),
            "solve_RpFinv": rel_res,
        }
        return loss, info

    def compute_precond(self, Up, log_F, sigma, ld_window, A):
        with torch.no_grad():
            Lam, V = torch.linalg.eigh(Up)
            mu = torch.exp(-log_F).mean()
            diag = sigma * (Lam + mu * sigma)
            P = V @ torch.diag(1 / diag) @ V.T
        return P


class WASPPreD(WASPPre):
    def compute_precond(self, Up, log_F, sigma, ld_window, A):
        with torch.no_grad():
            Fp = torch.diag(torch.exp(log_F))
            P_RmR = (1 / (sigma * sigma)) * Fp
        return P_RmR
