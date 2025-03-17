#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triangle inflow test with constant electric field
"""
import sys
import os
import shutil
import getopt
import numpy as np
from scipy import sparse
from scipy.io import savemat
from scipy.sparse.linalg import splu
import mfem.ser as mfem

from mfem_tensor_utils import mfem_sparse_to_csr, x1, x2,  \
    orthogonalize, tensor_diff, L2_norm, \
    euler_step, midpoint_step, rk3_ssp_step
from boltzmann_solution import BoltzmannSolution


#%% coefficients
one = mfem.ConstantCoefficient(1.0)


class chi_n(mfem.PyCoefficient):
    """MFEM Coefficient chi_{n*v<0}(v) * n * v"""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def EvalValue(self, v):
        z = self.n[0] * v[0] + self.n[1] * v[1]
        return z if z < 0 else 0.0


#%% settings

# default settings
run_id = 0  # id of run for output directory

refine = 0  # number of uniform refinements for spatial and velocity mesh

Tfinal = 0.5
dt_base = 5e-3  # base step size, dt = dt_base * 2**(-2*refine)
ode_step_id = 3  # ODE solver: Euler (1), Midpoint (2), RK3 SSP (3)

delta = 1e-2  # CIP stabilitzation parameter

tensorrank = 1
rauc_tau_base = 1e-4  # base tolerance for rank adaptivity: rauc_tau = rauc_tau_base * 2**(-2 * refine)
rauc_max_base = 20  # base maximal rank for adaptivity: rauc_max = (refine + 1) * rauc_max_base

sigmax = 0.2  # spatial width of initial condition
sigmav = 0.5  # velocity width of initial condition

bs_tau_base = 1e-5  # base tolerance for SVD of exact solution: bs_tau = bs_tau_base * 2**(-2 * refine)
bs_max = 100  # maximal rank for SVD of exact solution

visualization = False
do_wait = False

gridv_base = 64  # initial velocity mesh parameter

try:
    opts, args = getopt.getopt(
            sys.argv[1:],
            "",
            ["id=", "level=", "gridv_base=",
             "Tfinal=", "dt_base=", "ode_step=",
             "rank=", "rauc_tau_base=", "rauc_max_base=",
             "sigmax=", "sigmav=",
             "bs_tau_base=", "bs_max=",
             "glvis"])
except getopt.GetoptError:
    print(opts, args)
    print("wrong arguments")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--level"):
        refine = int(arg)
    elif opt in ("--gridv_base"):
        gridv_base = int(arg)
    elif opt in ("--id"):
        run_id = int(arg)
    elif opt in ("--Tfinal"):
        Tfinal = float(arg)
    elif opt in ("--dt_base"):
        dt = float(arg)
    elif opt in ("--ode_step"):
        ode_step_id = int(arg)
    elif opt in ("--rank"):
        tensorrank = int(arg)
    elif opt in ("--rauc_tau_base"):
        rauc_tau_base = float(arg)
    elif opt in ("--rauc_max_base"):
        rauc_max_base = int(arg)
    elif opt in ("--sigmax"):
        sigmax = float(arg)
    elif opt in ("--sigmav"):
        sigmav = float(arg)
    elif opt in ("--bs_tau_base"):
        bs_tau_base = float(arg)
    elif opt in ("--bs_max"):
        bs_max = int(arg)
    elif opt in ("--glvis"):
        visualization = True
    else:
        print("Error: options")
        sys.exit(2)

# derived
dt = dt_base * 2**(-2 * refine)
rauc_tau = rauc_tau_base * 2**(-2 * refine)
rauc_max = (refine + 1) * rauc_max_base
bs_tau = bs_tau_base * 2**(-2 * refine)
match ode_step_id:
    case 1:
        ode_step = euler_step
    case 2:
        ode_step = midpoint_step
    case 3:
        ode_step = rk3_ssp_step
    case _:
        print("Error: options")
        sys.exit(2)

# output directory
outdir = "triangle_inflow_out_%i" % run_id
try:
    os.makedirs(outdir)
except FileExistsError:
    if run_id == 0:
        # directory already exists => remove all old files
        shutil.rmtree(outdir)
        os.makedirs(outdir)
    else:
        print("Error: run exists - delete first!")
        print(f"rm -rf {outdir}")
        sys.exit(2)

fout = open("%s/out.txt" % outdir, "w")
fout.write("# triangle_inflow.py with command\n")
fout.write("# %s\n" % " ".join(sys.argv[1:]))
fout.write("# run_id = %i\n" % run_id)
fout.write("# refine = %i\n" % refine)
fout.write("# gridv_base = %i\n" % gridv_base)
fout.write("# Tfinal = %10.4f\n" % Tfinal)
fout.write("# dt = %10.5e\n" % dt)
fout.write("# ode_step_id = %i\n" % ode_step_id)
fout.write("# delta = %10.5e\n" % delta)
fout.write("# tensorrank = %i\n" % tensorrank)

fout.write("# rauc_tau = %10.5e\n" % rauc_tau)
fout.write("# rauc_max = %i\n" % rauc_max)

fout.write("# sigmax = %f\n" % sigmax)
fout.write("# sigmav = %f\n" % sigmav)

fout.write("# bs_tau = %10.5e\n" % bs_tau)
fout.write("# bs_max = %i\n" % bs_max)
fout.write("#%3s %20s %4s %20s %20s\n"
           % ("i", "t", "rank", "err L2", "nrm L2"))
fout.flush()


# meshfiles
meshfilex0 = "triangle_x_%i.mesh" % 0
meshfilev0 = "box_v_4.0_%i.mesh" % gridv_base


# list norm (normal, <list of edge id's>)
boundaries = [
    ([ 0.5/np.sqrt(1.25), -1.0/np.sqrt(1.25)], [1]),  # lower right
    ([ 0.5/np.sqrt(1.25), +1.0/np.sqrt(1.25)], [2]),  # upper right
    ([-1.0,                0.0],               [3]),  # left
    ]


do_inflow = True

# electrical field
E1_const = 0.0
E2_const = 4.0

# center of initial condition
x0_c = [-0.50-sigmax, 0.1]
v0_c = [+2.00, 0.0]

# boltzmann solution
Lx_bs = 0.5
Nx_bs = 40 * 2 ** refine
Lv_bs = 4.0
Nv_bs = 2 * gridv_base * 2 ** refine

bs = BoltzmannSolution(x0_c, sigmax, v0_c, sigmav,
                       [E1_const, E2_const],
                       Lx_bs, Nx_bs, Lv_bs, Nv_bs, bs_max, bs_tau)


#%% general setting time and plotting
nt = int(np.round(Tfinal/dt))
dt = Tfinal/nt

nplot = 20  # number of output checkpoints
plotmod = int(nt/nplot)


#%% x discretization
order = 1  # order of discretization

meshx = mfem.Mesh("data/%s" % meshfilex0, 1, 1)
# refinement
for i in range(refine):
    meshx.UniformRefinement()

fecx = mfem.H1_FECollection(order,  meshx.Dimension())
fespacex = mfem.FiniteElementSpace(meshx, fecx)
rhox_gf = mfem.GridFunction(fespacex)

# save mesh file
meshx.Print("%s/meshx.mesh" % outdir)

# mass
mx = mfem.BilinearForm(fespacex)
mx.AddDomainIntegrator(mfem.MassIntegrator(one))
mx.Assemble()
mx.Finalize()
Mx = mfem_sparse_to_csr(mx.SpMat())

# transport
tx1 = mfem.BilinearForm(fespacex)
tx1.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 0))
tx1.Assemble()
tx1.Finalize()
Tx1 = mfem_sparse_to_csr(tx1.SpMat())

tx2 = mfem.BilinearForm(fespacex)
tx2.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 1))
tx2.Assemble()
tx2.Finalize()
Tx2 = mfem_sparse_to_csr(tx2.SpMat())

# < x_i psi, psi>
rule = mfem.IntRules.Get(2, 4)
mx1 = mfem.BilinearForm(fespacex)
mx1.AddDomainIntegrator(mfem.MassIntegrator(x1(), rule))
mx1.Assemble()
mx1.Finalize()
Mx1 = mfem_sparse_to_csr(mx1.SpMat())

mx2 = mfem.BilinearForm(fespacex)
mx2.AddDomainIntegrator(mfem.MassIntegrator(x2(), rule))
mx2.Assemble()
mx2.Finalize()
Mx2 = mfem_sparse_to_csr(mx2.SpMat())

# boundary
Mbxs = []
nb = meshx.bdr_attributes.Max()

for i in range(len(boundaries)):
    bdr_marker = mfem.intArray(nb)
    bdr_marker.Assign(0)
    for k in boundaries[i][1]:
        bdr_marker[k - 1] = 1  # 1 based to 0 based index

    mbx = mfem.BilinearForm(fespacex)
    mbx.AddBoundaryIntegrator(mfem.MassIntegrator(one), bdr_marker)
    mbx.Assemble()
    mbx.Finalize()
    Mbxs.append(mfem_sparse_to_csr(mbx.SpMat()))

# electrical field
E1 = mfem.ConstantCoefficient(E1_const)
mE1 = mfem.BilinearForm(fespacex)
mE1.AddDomainIntegrator(mfem.MassIntegrator(E1))
mE1.Assemble()
mE1.Finalize()
ME1 = mfem_sparse_to_csr(mE1.SpMat())

E2 = mfem.ConstantCoefficient(E2_const)
mE2 = mfem.BilinearForm(fespacex)
mE2.AddDomainIntegrator(mfem.MassIntegrator(E2))
mE2.Assemble()
mE2.Finalize()
ME2 = mfem_sparse_to_csr(mE2.SpMat())

# stabilization
stabx = mfem.BilinearForm(fespacex)
stabx.AddInteriorFaceIntegrator(mfem.CIPTraceIntegrator())
stabx.Assemble()
stabx.Finalize()
Stabx = mfem_sparse_to_csr(stabx.SpMat())


#%% v discretization
order = 1

meshv = mfem.Mesh("data/%s" % meshfilev0, 1, 1)
# refinement
for i in range(refine):
    meshv.UniformRefinement()


fecv = mfem.H1_FECollection(order,  meshv.Dimension())
fespacev = mfem.FiniteElementSpace(meshv, fecv)
rhov_gf = mfem.GridFunction(fespacev)

# save mesh file
meshv.Print("%s/meshv.mesh" % outdir)

# mass
mv = mfem.BilinearForm(fespacev)
mv.AddDomainIntegrator(mfem.MassIntegrator(one))
mv.Assemble()
mv.Finalize()
Mv = mfem_sparse_to_csr(mv.SpMat())

# transport
tv1 = mfem.BilinearForm(fespacev)
tv1.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 0))
tv1.Assemble()
tv1.Finalize()
Tv1 = mfem_sparse_to_csr(tv1.SpMat())

tv2 = mfem.BilinearForm(fespacev)
tv2.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 1))
tv2.Assemble()
tv2.Finalize()
Tv2 = mfem_sparse_to_csr(tv2.SpMat())

# < v_i psi, psi>
rule = mfem.IntRules.Get(2, 4)
mv1 = mfem.BilinearForm(fespacev)
mv1.AddDomainIntegrator(mfem.MassIntegrator(x1(), rule))
mv1.Assemble()
mv1.Finalize()
Mv1 = mfem_sparse_to_csr(mv1.SpMat())

mv2 = mfem.BilinearForm(fespacev)
mv2.AddDomainIntegrator(mfem.MassIntegrator(x2(), rule))
mv2.Assemble()
mv2.Finalize()
Mv2 = mfem_sparse_to_csr(mv2.SpMat())


# boundary terms
Mbvs = []
for i in range(len(boundaries)):
    mbv = mfem.BilinearForm(fespacev)
    coeff = chi_n(boundaries[i][0])
    mbv.AddDomainIntegrator(mfem.MassIntegrator(coeff, rule))
    mbv.Assemble()
    mbv.Finalize()
    Mbvs.append(mfem_sparse_to_csr(mbv.SpMat()))

# stabilization
stabv = mfem.BilinearForm(fespacev)
stabv.AddInteriorFaceIntegrator(mfem.CIPTraceIntegrator())
stabv.Assemble()
stabv.Finalize()
Stabv = mfem_sparse_to_csr(stabv.SpMat())


#%% fine spaces for error evaluation, refined once
# x space
meshx_fine = mfem.Mesh("data/%s" % meshfilex0, 1, 1)
for i in range(refine):
    meshx_fine.UniformRefinement()
fespacex_fine = mfem.FiniteElementSpace(meshx_fine, fecx)
fespacex_fine.SetUpdateOperatorType(mfem.Operator.MFEM_SPARSEMAT)

meshx_fine.UniformRefinement()
fespacex_fine.Update(True)
op_handlex = mfem.OperatorHandle(mfem.Operator.MFEM_SPARSEMAT)
fespacex_fine.GetUpdateOperator(op_handlex)
Px = op_handlex.AsSparseMatrix()
Pmatx = mfem_sparse_to_csr(Px)

rhox_gf_fine = mfem.GridFunction(fespacex_fine)

# save mesh file
meshx_fine.Print("%s/meshx_fine.mesh" % outdir)

# v space
meshv_fine = mfem.Mesh("data/%s" % meshfilev0, 1, 1)
for i in range(refine):
    meshv_fine.UniformRefinement()
fespacev_fine = mfem.FiniteElementSpace(meshv_fine, fecv)
fespacev_fine.SetUpdateOperatorType(mfem.Operator.MFEM_SPARSEMAT)

meshv_fine.UniformRefinement()
fespacev_fine.Update(True)
op_handlev = mfem.OperatorHandle(mfem.Operator.MFEM_SPARSEMAT)
fespacev_fine.GetUpdateOperator(op_handlev)
Pv = op_handlev.AsSparseMatrix()
Pmatv = mfem_sparse_to_csr(Pv)

rhov_gf_fine = mfem.GridFunction(fespacev_fine)

# save mesh file
meshv_fine.Print("%s/meshv_fine.mesh" % outdir)

# mass matrices
mx_fine = mfem.BilinearForm(fespacex_fine)
mx_fine.AddDomainIntegrator(mfem.MassIntegrator(one))
mx_fine.Assemble()
mx_fine.Finalize()
Mx_fine = mfem_sparse_to_csr(mx_fine.SpMat())

mv_fine = mfem.BilinearForm(fespacev_fine)
mv_fine.AddDomainIntegrator(mfem.MassIntegrator(one))
mv_fine.Assemble()
mv_fine.Finalize()
Mv_fine = mfem_sparse_to_csr(mv_fine.SpMat())


#%% lu decomposition
Mxinv = splu(Mx)
Mvinv = splu(Mv)


#%% exact solution
class XV_bs_fun(mfem.PyCoefficient):
    """MFEM Coefficient wrapper for tensor representation of exact solution"""
    def __init__(self, bs, i, xv):
        """

        Parameters
        ----------
        bs  instance of BoltzmannSolution
        i   numer of factor
        xv  "x" or "v" for spatial or velocity values
        """
        super().__init__()
        self.bs = bs
        self.i = i
        self.xv = xv
        return

    def EvalValue(self, z):
        match self.xv:
            case "x":
                u = self.bs.eval_factorX(self.i, z)
            case "v":
                u = self.bs.eval_factorV(self.i, z)
            case _:
                raise Exception("Setting does not exist.")
        return u


def compute_bs_solution(bs, t, fine=False):
    """Return exact solution as FEM representation X S V^T"""
    bs.set_time(t)
    rank = bs.get_rank()

    if fine is False:
        nx = fespacex.GetNDofs()
        nv = fespacev.GetNDofs()
        X_grid = mfem.GridFunction(fespacex)
        V_grid = mfem.GridFunction(fespacev)
    else:
        nx = fespacex_fine.GetNDofs()
        nv = fespacev_fine.GetNDofs()
        X_grid = mfem.GridFunction(fespacex_fine)
        V_grid = mfem.GridFunction(fespacev_fine)

    X = np.zeros((nx, rank))
    V = np.zeros((nv, rank))
    S = np.zeros((rank, rank))

    for i in range(rank):
        X_grid.ProjectCoefficient(XV_bs_fun(bs, i, "x"))
        X[:, i] = X_grid.GetDataArray()

        V_grid.ProjectCoefficient(XV_bs_fun(bs, i, "v"))
        V[:, i] = V_grid.GetDataArray()

        S[i, i] = bs.eval_factorS(i)

    return X, S, V


class BsSolutionInterpolator:
    """Interpolate factor of exact solution on regular grid to FE function"""
    def __init__(self, bs, fespacex, fespacev):
        self.bs = bs

        self.nx = fespacex.GetNDofs()
        self.nv = fespacev.GetNDofs()

        self.Px1, self.Px2 = self.fespace_interpolation(fespacex, "x")
        self.Pv1, self.Pv2 = self.fespace_interpolation(fespacev, "v")
        return

    def compute(self):
        rank = bs.get_rank()

        X = np.zeros((self.nx, rank))
        V = np.zeros((self.nv, rank))
        S = np.zeros((rank, rank))

        for i in range(rank):
            r0, r1 = bs.get_ranks()
            i0 = i % r0
            i1 = i // r0

            X[:, i] = (self.Px1 @ self.bs.Qx[0][:, i0]) \
                * (self.Px2 @ self.bs.Qx[1][:, i1])
            V[:, i] = (self.Pv1 @ self.bs.Qv[0][:, i0]) \
                * (self.Pv2 @ self.bs.Qv[1][:, i1])
            S[i, i] = self.bs.eval_factorS(i)
        return X, S, V

    def fespace_interpolation(self, fespace, xy):
        match xy:
            case "x":
                z_source = self.bs.x
            case "v":
                z_source = self.bs.v
            case _:
                print("error")

        coords = mfem.GridFunction(fespace)
        coords.ProjectCoefficient(x1())
        z_dest = coords.GetDataArray()
        P1 = self.grid_interpolation(z_source, z_dest)

        coords.ProjectCoefficient(x2())
        z_dest = coords.GetDataArray()
        P2 = self.grid_interpolation(z_source, z_dest)

        return P1, P2

    def grid_interpolation(self, x_source, x_dest):

        n_source = len(x_source)
        n_dest = len(x_dest)

        PI = np.zeros(2*n_dest, dtype=np.int64)
        PJ = np.zeros(2*n_dest, dtype=np.int64)
        PV = np.zeros(2*n_dest)

        index_source = np.arange(n_source)

        for i in range(n_dest):
            k = np.interp(x_dest[i], x_source, index_source)
            k1 = int(np.floor(k))
            k2 = k1 + 1 if k1 + 1 < n_source else k1 - 1
            w1 = 1.0 - (k - k1)
            w2 = k - k1

            PI[2*i] = i
            PI[2*i+1] = i
            PJ[2*i] = k1
            PJ[2*i+1] = k2
            PV[2*i] = w1
            PV[2*i+1] = w2
        P = sparse.csr_matrix((PV, (PI, PJ)), shape=(n_dest, n_source))
        return P


bs_interpolator = BsSolutionInterpolator(bs, fespacex, fespacev)
bs_interpolator_fine = BsSolutionInterpolator(bs, fespacex_fine, fespacev_fine)


#%% visualization
if visualization:
    meshx.Save("out/rhox.mesh")
    meshv.Save("out/rhov.mesh")
    sol_sock = mfem.socketstream("localhost", 19916)
    sol_sock.precision(8)


def compute_rhox_gf(X, S, V, is_fine=False):
    """Compute spatial density by integrating out velocity variables"""
    if not is_fine:
        rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv @ V)
        rhox_gf.GetDataArray()[:] = (X @ S @ rhoV.T).reshape(-1)
    else:
        rhoV = np.ones((1, fespacev_fine.GetNDofs())) @ (Mv_fine @ V)
        rhox_gf_fine.GetDataArray()[:] = (X @ S @ rhoV.T).reshape(-1)
    return


def plot_rhoX(X, S, V, is_fine=False):
    """Plot spatial density via GLVis"""
    if not is_fine:
        compute_rhox_gf(X, S, V, is_fine=False)
        sol_sock.send_solution(meshx, rhox_gf)
    else:
        compute_rhox_gf(X, S, V, is_fine=True)
        sol_sock.send_solution(meshx_fine, rhox_gf_fine)
    return


def save_rhoX(X, S, V, i=0, outdir="out", name="", is_fine=False):
    """Save spatial density in GLVis format"""
    compute_rhox_gf(X, S, V, is_fine=is_fine)
    if not is_fine:
        rhox_gf.Save("%s/rhox%s_%4.4i.gf" % (outdir, name, i))
    else:
        rhox_gf_fine.Save("%s/rhox%s_%4.4i.gf" % (outdir, name, i))
    return


def compute_rhov_gf(X, S, V, is_fine=False):
    """Compute velocity density by integrating out spatial variables"""
    if not is_fine:
        rhoX = np.ones((1, fespacex.GetNDofs())) @ (Mx @ X)
        rhov_gf.GetDataArray()[:] = (V @ S.T @ rhoX.T).reshape(-1)
    else:
        rhoX = np.ones((1, fespacex_fine.GetNDofs())) @ (Mx_fine @ X)
        rhov_gf_fine.GetDataArray()[:] = (V @ S.T @ rhoX.T).reshape(-1)
    return


def plot_rhoV(X, S, V, is_fine=False):
    """Plot velocity density via GLVis"""
    if not is_fine:
        compute_rhov_gf(X, S, V, is_fine=False)
        sol_sock.send_solution(meshv, rhov_gf)
    else:
        compute_rhov_gf(X, S, V, is_fine=True)
        sol_sock.send_solution(meshv_fine, rhov_gf_fine)
    return


def save_rhoV(X, S, V, i=0, outdir="out", name="", is_fine=False):
    """Save velocity density in GLVis format"""
    compute_rhov_gf(X, S, V, is_fine=is_fine)
    if not is_fine:
        rhov_gf.Save("%s/rhov%s_%4.4i.gf" % (outdir, name, i))
    else:
        rhov_gf_fine.Save("%s/rhov%s_%4.4i.gf" % (outdir, name, i))
    return


#%% initial condition
X0_sol, S0_sol, V0_sol = compute_bs_solution(bs, 0.0)
crank = min(S0_sol.shape[0], tensorrank)

nx = fespacex.GetNDofs()
nv = fespacev.GetNDofs()

X0 = np.zeros((nx, tensorrank))
V0 = np.zeros((nv, tensorrank))
S0 = np.zeros((tensorrank, tensorrank))

X0[:, :crank] = X0_sol[:, :crank]
V0[:, :crank] = V0_sol[:, :crank]
S0[:crank, :crank] = S0_sol[:crank, :crank]

errmax = 0.0

#%% orthonormalize initial condition
X0, Sx = orthogonalize(X0, Mx)
V0, Sv = orthogonalize(V0, Mv)
S0 = Sx @ S0 @ Sv.T


#%% step function: first order adaptive with unconventional integrator
def step1(X, S, V, dt, gb=None, Xb=None, Sb=None, Vb=None):
    """Single time step for dynmical low rank approximation"""

    def eval_rhsK(tau, K, V):
        rhsK = (
            - (Tx1 @ K) @ (V.T @ (Mv1 @ V)).T
            - (Tx2 @ K) @ (V.T @ (Mv2 @ V)).T
            + (ME1 @ K) @ (V.T @ (Tv1 @ V)).T
            + (ME2 @ K) @ (V.T @ (Tv2 @ V)).T)
        if delta != 0.0:
            rhsK -= delta * (Stabx @ K)

        for i in range(len(Mbxs)):
            rhsK += (Mbxs[i] @ K) @ (V.T @ (Mbvs[i] @ V)).T
            if Xb is not None:
                for j in range(len(Xb)):
                    rhsK += -gb[j](tau) * ((Mbxs[i] @ (Xb[j] @ Sb[j])) @ (V.T @ (Mbvs[i] @ Vb[j])).T)

        dK = Mxinv.solve(rhsK)
        return dK

    def eval_rhsS(tau, X, S, V):
        dS = (
            - (X.T @ (Tx1 @ X)) @ S @ (V.T @ (Mv1 @ V)).T
            - (X.T @ (Tx2 @ X)) @ S @ (V.T @ (Mv2 @ V)).T
            + (X.T @ (ME1 @ X)) @ S @ (V.T @ (Tv1 @ V)).T
            + (X.T @ (ME2 @ X)) @ S @ (V.T @ (Tv2 @ V)).T)

        for i in range(len(Mbxs)):
            dS += (X.T @ (Mbxs[i] @ X)) @ S @ (V.T @ (Mbvs[i] @ V)).T
            if Xb is not None:
                for j in range(len(Xb)):
                    dS += -gb[j](tau) * ((X.T @ (Mbxs[i] @ Xb[j])) @ Sb[j] @ (V.T @ (Mbvs[i] @ Vb[j])).T)
        return dS

    def eval_rhsL(tau, L, X):
        rhsL = (
            - (Mv1 @ L) @ (X.T @ (Tx1 @ X)).T
            - (Mv2 @ L) @ (X.T @ (Tx2 @ X)).T
            + (Tv1 @ L) @ (X.T @ (ME1 @ X)).T
            + (Tv2 @ L) @ (X.T @ (ME2 @ X)).T)

        if delta != 0.0:
            rhsL -= delta * (Stabv @ L)

        for i in range(len(Mbxs)):
            rhsL += (Mbvs[i] @ L) @ (X.T @ (Mbxs[i] @ X)).T
            if Xb is not None:
                for j in range(len(Xb)):
                    rhsL += -gb[j](tau) * ((Mbvs[i] @ (Vb[j] @ Sb[j].T)) @ (X.T @ (Mbxs[i] @ Xb[j])).T)

        dL = Mvinv.solve(rhsL)
        return dL

    # start
    currank = X.shape[1]

    # STEP K +dt with ic X,S,V for 0-> dt
    K = X @ S
    K1 = ode_step(lambda tau, Y: eval_rhsK(tau, Y, V), 0.0, dt, K)
    Xhat, Sx = orthogonalize(np.hstack((X, K1)), Mx)
    Rx = Sx[:, :currank]

    # STEP L +dt with ic X,S,V for 0-> dt
    L = V @ S.T
    L1 = ode_step(lambda tau, Y: eval_rhsL(tau, Y, X), 0.0, dt, L)
    Vhat, Sv = orthogonalize(np.hstack((V, L1)), Mv)
    Rv = Sv[:, :currank]

    # STEP S +dt with ic Xhat,S,Vhat for 0 -> dt
    Shat = Rx @ S @ Rv.T
    Shat1 = ode_step(lambda tau, Y: eval_rhsS(tau, Xhat, Y, Vhat),
                     0.0, dt, Shat)

    # truncate Shat
    Qx, sigma, QvT = np.linalg.svd(Shat1)
    Qv = QvT.T

    # sum up squares of singular values from small to big and test
    # against tolerance
    # kinv is the numer of sing. values that can be left out
    sigma_flipped = np.flip(sigma)
    mask = np.cumsum(sigma_flipped**2) < rauc_tau**2
    kinv = np.argmin(mask)
    if mask[kinv] == True:  # all values smaller than rauc_tau
        kinv = 2 * currank


    newrank = max(1, min(2*currank - kinv, rauc_max))

    # cut down rank
    S = np.diag(sigma[:newrank])
    X = Xhat @ Qx[:, :newrank]
    V = Vhat @ Qv[:, :newrank]

    return (X, S, V)


#%% time stepping
X = X0.copy()
S = S0.copy()
V = V0.copy()

iplot = 0
save_rhoX(X, S, V, iplot, outdir=outdir)
save_rhoV(X, S, V, iplot, outdir=outdir)
savemat("%s/step_%4.4i.mat" % (outdir, iplot), {"X": X, "S": S, "V": V})
iplot += 1

t = 0.0

for i in range(nt):
    bs.set_time((i + 0.5) * dt)  # compute solution at middle of current time interval
    Xb, Sb, Vb = bs_interpolator.compute()

    (X, S, V) = step1(X, S, V, dt,
                      gb=[lambda t: 1.0], Xb=[Xb], Sb=[Sb], Vb=[Vb])

    t = (i+1) * dt
    print(f"{i+1}/{nt}: {t:10.6f} rank={S.shape[0]}")

    # plot
    if (i + 1) % plotmod == 0:
        print("---> output")

        nrm0 = L2_norm(X, S, V, Mx, Mv)
        bs.set_time(t)
        X_sol_fine, S_sol_fine, V_sol_fine = bs_interpolator_fine.compute()

        dX, dS, dV = tensor_diff(Pmatx @ X, S, Pmatv @ V,
                                 X_sol_fine, S_sol_fine, V_sol_fine)
        nrmd = L2_norm(dX, dS, dV, Mx_fine, Mv_fine)
        errmax = max(errmax, nrmd)
        print(f"err(fine) = {nrmd : 10.2e} of {nrm0 : 10.2e} (maximal = {errmax : 10.2e})")

        save_rhoX(X_sol_fine, S_sol_fine, V_sol_fine, iplot,
                  outdir=outdir, name="_sol", is_fine=True)
        save_rhoV(X_sol_fine, S_sol_fine, V_sol_fine, iplot,
                  outdir=outdir, name="_sol", is_fine=True)

        save_rhoX(dX, dS, dV, iplot, outdir=outdir, name="_err",
                  is_fine=True)
        save_rhoV(dX, dS, dV, iplot, outdir=outdir, name="_err",
                  is_fine=True)

        if fout:
            fout.write("%4i %20.10e %4i %20.10e %20.10e\n"
                       % (iplot, t, S.shape[0], nrmd, nrm0))
            fout.flush()

        # save densities
        save_rhoX(X, S, V, iplot, outdir=outdir)
        save_rhoV(X, S, V, iplot, outdir=outdir)
        savemat("%s/step_%4.4i.mat" % (outdir, iplot),
                {"X": X, "S": S, "V": V})
        iplot += 1

        if visualization:
            plot_rhoX(X, S, V)
            # plot_rhoV(X, S, V)
            if do_wait:
                input("Press Enter to continue...")


savemat("%s/step.mat" % outdir, {"X": X, "S": S, "V": V})
