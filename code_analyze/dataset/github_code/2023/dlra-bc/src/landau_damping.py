#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landau Damping 2+2d
"""
import sys
import os
import shutil
import getopt
import numpy as np
from scipy.io import savemat
from scipy.sparse.linalg import splu
import mfem.ser as mfem

from mfem_tensor_utils import mfem_sparse_to_csr, x1, x2, sum_squared, \
    orthogonalize, L2_norm, \
    euler_step, midpoint_step, rk3_ssp_step
from electric_field_landau import ElectricFieldLandau


#%% coefficients
one = mfem.ConstantCoefficient(1.0)

#%% settings

# default settings
run_id = 0  # id of run for output directory

refine = 0  # number of uniform refinements for spatial and velocity mesh

Tfinal = 50
dt = 1e-3  # step size

ode_step_id = 1  # ODE solver: Euler (1), Midpoint (2), RK3 SSP (3)

delta = 1e-2  # CIP stabilitzation parameter

tensorrank = 3

time_method = 2  # 1: first order time stepping, 2: RAUC

rauc_tau = 1e-4  # tolerance for rank adaptivity
rauc_max = 10  # maximal rank for adaptivity

visualization = False

gridx_base = 64
gridv_base = 256

# input argument parsing
try:
    opts, args = getopt.getopt(
            sys.argv[1:],
            "",
            ["id=", "level=", "gridx_base=", "gridv_base=",
             "Tfinal=", "dt=", "ode_step=", "delta=",
             "rank=", "time_method=", "rauc_tau=", "rauc_max=",
             "glvis", "new"])
except getopt.GetoptError:
    print("wrong arguments")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--level"):
        refine = int(arg)
    elif opt in ("--gridv_base"):
        gridv_base = int(arg)
    elif opt in ("--gridx_base"):
        gridx_base = int(arg)
    elif opt in ("--id"):
        run_id = int(arg)
    elif opt in ("--Tfinal"):
        Tfinal = float(arg)
    elif opt in ("--delta"):
        delta = float(arg)
    elif opt in ("--dt"):
        dt = float(arg)
    elif opt in ("--ode_step"):
        ode_step_id = int(arg)
    elif opt in ("--time_method"):
        time_method = int(arg)
    elif opt in ("--rank"):
        tensorrank = int(arg)
    elif opt in ("--rauc_tau"):
        rauc_tau = float(arg)
    elif opt in ("--rauc_max"):
        rauc_max = int(arg)
    elif opt in ("--glvis"):
        visualization = True
    else:
        print("Error: options")
        sys.exit(2)

# derived
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
outdir = "landau_damping_out_%i" % run_id
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
fout.write("# landau damping 2+2d with command\n")
fout.write("# %s\n" % " ".join(sys.argv[1:]))
fout.write("# run_id = %i\n" % run_id)
fout.write("# refine = %i\n" % refine)
fout.write("# gridx_base = %i\n" % gridx_base)
fout.write("# gridv_base = %i\n" % gridv_base)
fout.write("# Tfinal = %10.4f\n" % Tfinal)
fout.write("# dt = %10.5e\n" % dt)
fout.write("# ode_step_id = %i\n" % ode_step_id)
fout.write("# delta = %10.5e\n" % delta)
fout.write("# tensorrank = %i\n" % tensorrank)
fout.write("# time_method = %i\n" % time_method)
fout.write("# rauc_tau = %10.5e\n" % rauc_tau)
fout.write("# rauc_max = %i\n" % rauc_max)

# header for output of physical invariants
fout.write("#%9s %6s %20s %20s %20s %20s %20s %20s %20s\n"
           % ("t", "rank", "el. energy",
              "mass", "rel. err. mass",
              "E_total", "rel. err. E_total",
              "entropy", "rel. err. entropy"))
fout.flush()


# mesh files
meshfilex0 = "landau_per_x_%i.mesh" % gridx_base
meshfilev0 = "box_v_6.0_%i.mesh" % gridv_base

# time integration
nt = int(np.round(Tfinal/dt))
dt = Tfinal/nt

# output
do_wait = False  # True/False
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

# electrical field
efield_solver = ElectricFieldLandau(meshx, 1)

# integration rules
int_order = order + 1
irs = [mfem.IntRules.Get(i, int_order) for i in range(mfem.Geometry.NumGeom)]

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

# < |v| 2 psi, psi>
rule = mfem.IntRules.Get(2, 4)  # geometry, order=4
mv_squared = mfem.BilinearForm(fespacev)
mv_squared.AddDomainIntegrator(mfem.MassIntegrator(sum_squared(), rule))
mv_squared.Assemble()
mv_squared.Finalize()
Mv_squared = mfem_sparse_to_csr(mv_squared.SpMat())


# stabilization
stabv = mfem.BilinearForm(fespacev)
stabv.AddInteriorFaceIntegrator(mfem.CIPTraceIntegrator())
stabv.Assemble()
stabv.Finalize()
Stabv = mfem_sparse_to_csr(stabv.SpMat())


#%% lu decomposition
Mxinv = splu(Mx)
Mvinv = splu(Mv)


#%% visualization
if visualization:
    sol_sock = mfem.socketstream("localhost", 19916)
    sol_sock.precision(8)


def compute_rhox_gf(X, S, V):
    """Compute spatial density by integrating out velocity variables"""
    rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv @ V)
    rhox_gf.GetDataArray()[:] = (X @ S @ rhoV.T).reshape(-1)
    return


def plot_rhoX(X, S, V):
    """Plot spatial density via GLVis"""
    compute_rhox_gf(X, S, V)
    sol_sock.send_solution(meshx, rhox_gf)
    return


def save_rhoX(X, S, V, i=0, outdir="out", name=""):
    """Save spatial density in GLVis format"""
    compute_rhox_gf(X, S, V)
    rhox_gf.Save("%s/rhox%s_%4.4i.gf" % (outdir, name, i))
    return


def compute_rhov_gf(X, S, V):
    """Compute velocity density by integrating out spatial variables"""
    rhoX = np.ones((1, fespacex.GetNDofs())) @ (Mx @ X)
    rhov_gf.GetDataArray()[:] = (V @ S.T @ rhoX.T).reshape(-1)
    return


def plot_rhoV(X, S, V):
    """Plot velocity density via GLVis"""
    compute_rhov_gf(X, S, V)
    sol_sock.send_solution(meshv, rhov_gf)
    return


def save_rhoV(X, S, V, i=0, outdir="out", name=""):
    """Save velocity density in GLVis format"""
    compute_rhov_gf(X, S, V)
    rhov_gf.Save("%s/rhov%s_%4.4i.gf" % (outdir, name, i))
    return


#%% interpolation of initial conditions
class X0_fun(mfem.PyCoefficient):
    """Spatial initial value as MFEM coefficient"""
    def __init__(self):
        super().__init__()
        self.k = 0.5
        self.alpha = 1e-2

    def EvalValue(self, x):
        return 1.0 \
            + self.alpha * np.cos(self.k * x[0]) \
            + self.alpha * np.cos(self.k * x[1])


class V0_fun(mfem.PyCoefficient):
    """Velocity initial value as MFEM coefficient"""
    def EvalValue(self, v):
        return np.exp(-0.5 * (v[0]**2 + v[1]**2)) / (2.0 * np.pi)


X0_grid = mfem.GridFunction(fespacex)
X0_grid.ProjectCoefficient(X0_fun())
X0_1 = X0_grid.GetDataArray().copy()

nx = fespacex.GetNDofs()
X0 = np.zeros((nx, tensorrank))
X0[:, 0] = X0_1


V0_grid = mfem.GridFunction(fespacev)
V0_grid.ProjectCoefficient(V0_fun())
V0_1 = V0_grid.GetDataArray().copy()

nv = fespacev.GetNDofs()
V0 = np.zeros((nv, tensorrank))
V0[:, 0] = V0_1

S0 = np.zeros((tensorrank, tensorrank))
S0[0, 0] = 1.0


#%% orthonormalize initial condition
X0, Sx = orthogonalize(X0, Mx)
V0, Sv = orthogonalize(V0, Mv)
S0 = Sx @ S0 @ Sv.T


#%% step function

def update_E(X, S, V):
    """Compute electrical field corresponding to current density"""
    # print("E field calculation")
    global ME1, ME2
    # compute density
    rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv @ V)
    rhox_gf.GetDataArray()[:] = (X @ S @ rhoV.T).reshape(-1)
    rhox_gf_coeff = mfem.GridFunctionCoefficient(rhox_gf)

    # solve for efield
    phi, E1_gf, E2_gf = efield_solver.solve(rhox_gf_coeff)

    # compute electrical energy as squared L2 norm of coefficients
    E1_data = E1_gf.GetDataArray()
    E2_data = E2_gf.GetDataArray()
    E_electric = 0.5 * (E1_data @ (efield_solver.Mx_L2 @ E1_data)
                        + E2_data @ (efield_solver.Mx_L2 @ E2_data))

    # compute discretization matrices
    E1 = mfem.GridFunctionCoefficient(E1_gf)
    mE1 = mfem.BilinearForm(fespacex)
    mE1.AddDomainIntegrator(mfem.MassIntegrator(E1))
    mE1.Assemble()
    mE1.Finalize()
    ME1 = mfem_sparse_to_csr(mE1.SpMat())

    E2 = mfem.GridFunctionCoefficient(E2_gf)
    mE2 = mfem.BilinearForm(fespacex)
    mE2.AddDomainIntegrator(mfem.MassIntegrator(E2))
    mE2.Assemble()
    mE2.Finalize()
    ME2 = mfem_sparse_to_csr(mE2.SpMat())

    return E_electric

#%% step function
def step1(X, S, V, dt):
    """Single time step for dynmical low rank approximation"""

    def eval_rhsK(tau, K, V):
        global ME1, ME2
        rhsK = (
            - (Tx1 @ K) @ (V.T @ (Mv1 @ V)).T
            - (Tx2 @ K) @ (V.T @ (Mv2 @ V)).T
            + (ME1 @ K) @ (V.T @ (Tv1 @ V)).T
            + (ME2 @ K) @ (V.T @ (Tv2 @ V)).T)
        if delta != 0.0:
            rhsK -= delta * (Stabx @ K)

        dK = Mxinv.solve(rhsK)
        return dK

    def eval_rhsS(tau, X, S, V):
        global ME1, ME2
        dS = (
            - (X.T @ (Tx1 @ X)) @ S @ (V.T @ (Mv1 @ V)).T
            - (X.T @ (Tx2 @ X)) @ S @ (V.T @ (Mv2 @ V)).T
            + (X.T @ (ME1 @ X)) @ S @ (V.T @ (Tv1 @ V)).T
            + (X.T @ (ME2 @ X)) @ S @ (V.T @ (Tv2 @ V)).T)

        return dS

    def eval_rhsL(tau, L, X):
        global ME1, ME2
        rhsL = (
            - (Mv1 @ L) @ (X.T @ (Tx1 @ X)).T
            - (Mv2 @ L) @ (X.T @ (Tx2 @ X)).T
            + (Tv1 @ L) @ (X.T @ (ME1 @ X)).T
            + (Tv2 @ L) @ (X.T @ (ME2 @ X)).T)

        if delta != 0.0:
            rhsL -= delta * (Stabv @ L)

        dL = Mvinv.solve(rhsL)
        return dL


    #%% core step
    match time_method:
        #%% first order time splitting
        case 1:
            # STEP K +dt with ic X,S,V for 0-> dt
            K = X @ S
            K = ode_step(lambda tau, Y: eval_rhsK(tau, Y, V), 0.0, dt, K)
            X, S = orthogonalize(K, Mx)

            # STEP S +dt with ic X,S,V for 0 -> dt
            S = ode_step(lambda tau, Y: -eval_rhsS(tau, X, Y, V), 0.0, dt, S)

            # STEP L +dt with ic X,S,V for 0-> dt
            L = V @ S.T
            L = ode_step(lambda tau, Y: eval_rhsL(tau, Y, X), 0.0, dt, L)
            V, S = orthogonalize(L, Mv)
            S = S.T

        case 2:
            #%% first order adaptive with unconventional integrator
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
            mask = np.cumsum(sigma_flipped ** 2) < rauc_tau ** 2
            kinv = np.argmin(mask)
            if mask[kinv] == True:  ## all values smaller than rauc_tau
                kinv = 2 * currank
            newrank = max(1, min(2 * currank - kinv, rauc_max))

            # cut down rank
            S = np.diag(sigma[:newrank])
            X = Xhat @ Qx[:, :newrank]
            V = Vhat @ Qv[:, :newrank]

    return (X, S, V)


#%% physical quantities
def physical_quantities(X, S, V, E_electric):
    # mass
    rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv @ V)
    mass = np.sum(Mx @ (X @ S @ rhoV.T).reshape(-1))

    # kinetic energy
    rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv_squared @ V)
    E_kinetic = 0.5 * np.sum(Mx @ (X @ S @ rhoV.T).reshape(-1))

    # total energy
    E_total = E_kinetic + E_electric

    entropy = L2_norm(X, S, V, Mx, Mv)**2
    return mass, E_total, entropy


#%% time stepping
X = X0.copy()
S = S0.copy()
V = V0.copy()


iplot = 0
savemat("%s/step_%4.4i.mat" % (outdir, iplot), {"X": X, "S": S, "V": V})
iplot += 1

t = 0.0

# update e field
E_electric = update_E(X, S, V)

# compute physical quantities and write to file
mass_0, E_total_0, entropy_0 = physical_quantities(X, S, V, E_electric)
fout.write("%10.4f %6i %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e\n"
           % (t, S.shape[0], E_electric,
              mass_0, 0.0,
              E_total_0, 0.0,
              entropy_0, 0.0))
fout.flush()

# invariants
print(f"0/{nt}: {t:10.6f} rank={S.shape[0]} |E|={E_electric:10.2e}")

for i in range(nt):
    (X, S, V) = step1(X, S, V, dt)
    E_electric = update_E(X, S, V)
    t = (i+1) * dt

    # physical quantities
    mass, E_total, entropy = physical_quantities(X, S, V, E_electric)
    err_mass = np.abs((mass-mass_0)/mass_0)
    err_E_total = np.abs((E_total-E_total_0)/E_total_0)
    err_entropy = np.abs((entropy-entropy_0)/entropy_0)
    fout.write("%10.4f %6i %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e\n"
               % (t, S.shape[0], E_electric,
                  mass, err_mass,
                  E_total, err_E_total,
                  entropy, err_entropy))
    fout.flush()

    print(f"{i+1}/{nt}: {t:10.6f} rank={S.shape[0]} |E|={E_electric:10.2e} " 
          f"m: {err_mass:10.2e} E: {err_E_total:10.2e} S: {err_entropy:10.2e}")

    # save state
    if (i + 1) % plotmod == 0:
        state_filename = "%s/step_%4.4i.mat" % (outdir, iplot) 
        print("---> save state into %s" % state_filename)

        # save densities
        savemat(state_filename, {"X": X, "S": S, "V": V})
        iplot += 1
