# pylint: disable=C0103, R0902, R0903
"""Library to solve the time dependent schrodinger equation in the many-body Fock space.

This module contains the core computations
"""
from typing import Dict, Any, Tuple, List

import hashlib
import pickle

from numpy import ndarray
import numpy as np
from numpy.linalg import eigh

from scipy.integrate import solve_ivp
from scipy.linalg import logm
from scipy import sparse as sp

from random import normalvariate as rnormal

from numba import jit

from qlp.tdse.schedule import AnnealSchedule

from qlpdb.graph.models import Graph
from qlpdb.tdse.models import Tdse

from django.core.files.base import ContentFile
from django.conf import settings


def _set_up_pauli():
    """Creates Pauli matrices and identity
    """
    sigx = np.zeros((2, 2))
    sigz = np.zeros((2, 2))
    id2 = np.identity(2)
    proj0 = np.zeros((2, 2))
    proj1 = np.zeros((2, 2))
    sigplus = np.zeros((2, 2))
    sigminus = np.zeros((2, 2))
    sigx[0, 1] = 1.0
    sigx[1, 0] = 1.0
    sigz[0, 0] = 1.0

    sigz[1, 1] = -1.0
    proj0[0, 0] = 1.0
    proj1[1, 1] = 1.0
    sigplus[1, 0] = 1.0
    sigminus[0, 1] = 1.0
    return id2, sigx, sigz, proj0, proj1, sigplus, sigminus


ID2, SIG_X, SIG_Z, PROJ_0, PROJ_1, SIG_PLUS, SIG_MINUS = _set_up_pauli()


class PureSolutionInterface:
    """Interface for a pure state solution

    Attributes:
        t: array
        y: array
    """

    def __init__(self, y1):
        self.t = np.zeros((0))
        self.y = np.zeros((y1.size, 0))


def convert_params(params):
    for key in params:
        if key in ["hi_for_offset", "hi"]:
            params[key] = list(params[key])
        elif key in ["Jij"]:
            params[key] = [list(row) for row in params["Jij"]]
    return params


def get_or_create(Model, save=False, **kwargs):
    if save:
        obj, _ = Model.objects.get_or_create(**kwargs)
    else:
        try:
            obj = Model.objects.get(**kwargs)
        except Model.DoesNotExist:
            obj = Model(**dict((k, v) for (k, v) in kwargs.items() if "__" not in k))
    return obj


def save_file(query, instance, solution, filename, save=False):
    if save:
        content = pickle.dumps(instance)
        fid = ContentFile(content)
        query.instance.save(filename, fid)
        fid.close()
        content = pickle.dumps(solution)
        fid = ContentFile(content)
        query.solution.save(filename, fid)
        fid.close()
    else:
        instance_loc = f"{settings.MEDIA_ROOT}/temp/{filename}.instance"
        query.instance = instance_loc
        with open(instance_loc, "wb") as file:
            pickle.dump(instance, file)
        solution_loc = f"{settings.MEDIA_ROOT}/temp/{filename}.solution"
        query.solution = solution_loc
        with open(solution_loc, "wb") as file:
            pickle.dump(solution, file)


def add_jchaos(Jij_exact, hi_exact, jchaos):
    Jij = np.array(Jij_exact)
    hi = np.array(hi_exact)
    for i, Ji in enumerate(Jij):
        for j, J in enumerate(Ji):
            if Jij[i, j] == 0:
                pass
            else:
                Jij[i, j] += rnormal(0, jchaos * Jij[i, j])

    for i, h in enumerate(hi):
        if hi[j] == 0:
            pass
        else:
            hi[i] += rnormal(0, jchaos)
    return Jij, hi


class TDSE:
    """Time dependent Schrödinger equation solver class

    .. code-block:: python

        tdse = TDSE(n, ising_params, offset_params, solver_params)

        # Get thermodynamical state density
        temperature=15e-3
        initial_wavefunction = "true"
        rho = tdse.init_densitymatrix(temperature, initial_wavefunction)

        # Compute anneal
        sol_densitymatrix = tdse.solve_mixed(rho)
    """

    def __init__(
            self,
            graph_params: Dict[str, Any],
            ising_params: Dict[str, Any],
            offset_params: Dict[str, Any],
            solver_params: Dict[str, Any],
    ):
        """Init the class with

        Arguments:
            graph_params: Parameters of input graph
            ising_params: Parameters for the ising model, e.g., keys are
                {"Jij", "hi", "c", "energyscale"}.
            offset_params: Parameters for AnnealSchedule
            solver_params: Parameters for solve_ivp
        """
        self.graph = graph_params
        self.ising = ising_params
        self.offset = offset_params
        self.solver = solver_params
        (
            self.FockX,
            self.FockZ,
            self.FockZZ,
            self.Fockproj0,
            self.Fockproj1,
            self.Fockplus,
            self.Fockminus
        ) = self._init_Fock()
        self.Focksize = None
        self.AS = AnnealSchedule(**offset_params, graph_params=graph_params)
        self.IsingH = self._constructIsingH(
            self._Bij(self.AS.B(1)) * self.ising["Jij"], self.AS.B(1) * self.ising["hi"]
        )
        self.IsingH_exact = self._constructIsingH(np.array(self.ising["Jij"]), self.ising["hi"])
        self.gammadict = {"g": [], "glocal": [], "s": []}

    def hash_dict(self, d):
        hash = hashlib.md5(
            str([[key, d[key]] for key in sorted(d)]).replace(" ", "").encode("utf-8")
        ).hexdigest()
        return hash

    def summary(
            self, wave_params, instance, solution, time, probability, save=False,
    ):
        """
        output dictionary used to store tdse run into EspressodB

        tag: user-defined string (e.g. NN(3)_negbinary_-0.5_1.0_mixed_
        penalty: strength of penalty for slack variables
        ising_params: Jij, hi, c, energyscale (unit conversion from GHz)
        solver_params: method, rtol, atol
        offset_params:normalized_time, offset, hi_for_offset, offset_min, offset_range, fill_value, anneal_curve
        wave_params: pure or mixed, temp, initial_wavefunction. If pure, temp = 0
        instance: instance of tdse class
        time: solver time step
        probability: prob of Ising ground state
        """
        # make tdse inputs
        tdse_params = dict()
        wf_type = wave_params["type"]
        a_time = self.offset["annealing_time"]
        offset_type = self.offset["offset"]
        omin = self.offset["offset_min"]
        orange = self.offset["offset_range"]
        tdse_params["tag"] = f"{wf_type}_{a_time}us_{offset_type}_{omin}_{orange}"
        ising = dict(self.ising)
        ising["Jij"] = [list(row) for row in ising["Jij"]]
        ising["hi"] = list(ising["hi"])
        #ising["Jij_exact"] = [list(row) for row in ising["Jij_exact"]]
        #ising["hi_exact"] = list(ising["hi_exact"])
        tdse_params["ising"] = ising
        tdse_params["ising_hash"] = self.hash_dict(tdse_params["ising"])
        offset = dict(self.offset)
        offset["hi_for_offset"] = list(offset["hi_for_offset"])
        tdse_params["offset"] = offset
        tdse_params["offset_hash"] = self.hash_dict(tdse_params["offset"])
        solver = dict(self.solver)
        tdse_params["solver"] = solver
        tdse_params["solver_hash"] = self.hash_dict(tdse_params["solver"])
        tdse_params["wave"] = wave_params
        tdse_params["wave_hash"] = self.hash_dict(tdse_params["wave"])
        tdse_params["time"] = list(time)
        tdse_params["prob"] = list(probability)

        # select or insert row in graph
        gp = {key: self.graph[key] for key in self.graph if key not in ["total_qubits"]}
        graph = get_or_create(Model=Graph, save=save, **gp)
        tdse_params["graph"] = graph
        # select or insert row in tdse
        tdse = get_or_create(Model=Tdse, save=save, **tdse_params)
        # save pickled class instance
        tdsehash = self.hash_dict(
            {
                "ising": tdse_params["ising_hash"],
                "offset": tdse_params["offset_hash"],
                "solver": tdse_params["solver_hash"],
                "wave": tdse_params["wave_hash"],
            }
        )
        save_file(
            query=tdse,
            instance=instance,
            solution=solution,
            filename=tdsehash,
            save=save,
        )
        return tdse

    def _apply_H(self, t, psi: ndarray) -> ndarray:
        """Computes `i H(t) psi`"""
        return -1j * self.annealingH(t) @ psi

    def ground_state_degeneracy(
            self, H: ndarray, degeneracy_tol: float = 1e-6, debug: bool = False
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Computes the number of degenerate ground states

        Identifies degeneracy by comparing closeness to smallest eigenvalule.

        Arguments:
            H: Hamiltonian to compute eigen vectors off
            degeneracy_tol: Precision of comparison to GS
            debug: More output

        Returns: Ids for gs vectors, all eigenvalues, all eigenvectors
        """
        eigval, eigv = eigh(H.toarray())
        #mask = [abs((ei - eigval[0]) / eigval[0]) < degeneracy_tol for ei in eigval]
        gs_idx = [0, 1] #np.arange(len(eigval))[mask]
        if debug:
            print(
                f"Num. degenerate states @"
                f" s={self.offset['normalized_time'][1]}: {len(gs_idx)}"
            )
        return gs_idx, eigval, eigv

    @staticmethod
    def calculate_overlap(psi1: ndarray, psi2: ndarray, degen_idx: List[int]) -> float:
        """Computes overlaps of states in psi1 with psi2 (can be multiple)

        Overlap is defined as ``sum(<psi1_i | psi2>, i in degen_idx)``. This allows to
        compute overlap in presence of degeneracy. See als eq. (57) in the notes.

        Arguments:
            psi1: Set of wave vectors. Shape is (size_vector, n_vectors).
            psi2: Vector to compare against.
            degen_idx: Index for psi1 vectors.

        """
        return sum(
            np.absolute([psi1[:, idx].conj().dot(psi2) for idx in degen_idx]) ** 2
        )

    def init_eigen(self, dtype: str) -> Tuple[ndarray, ndarray]:
        """Computes eigenvalue and vector of initial Hamiltonian either as a pure
        eigenstate of `H_init` (transverse) or a s a superposition of
        `A(s) H_init + B(s) H_final` (true).
        """
        if dtype == "true":
            # true ground state
            eigvalue, eigvector = eigh(
                ((self.annealingH(s=self.offset["normalized_time"][0])).toarray()) / (self.ising["energyscale"])
            )
        elif dtype == "transverse":
            # DWave initial wave function
            # Ghz
            eigvalue, eigvector = eigh((
                                               -1
                                               * self._constructtransverseH(
                                           self.AS.A(0) * np.ones(self.graph["total_qubits"])
                                       )
                                       ).toarray())
        else:
            raise TypeError("Undefined initial wavefunction.")
        return eigvalue, eigvector

    def init_wavefunction(self, dtype="transverse") -> ndarray:
        """Returns wave function for first eigenstate of Hamiltonian of dtype.
        """
        _, eigvector = self.init_eigen(dtype)
        return (1.0 + 0.0j) * np.array(eigvector[:, 0]).flatten()

    def init_densitymatrix(
            self, temp: float = 13e-3, temp_local: float = 13e-3, dtype: str = "transverse", debug: bool = False
    ) -> ndarray:
        """Returns density matrix for s=0

        ``rho(s=0) = exp(- beta H(0)) / Tr(exp(- beta H(0)))``


        Arguments:
            temp: Temperature in K
            dtype: Kind of inital wave function (true or transverse)
            debug: More output messages
        """
        kb = 8.617333262145e-5  # Boltzmann constant [eV / K]
        # h = 4.135667696e-15  # Plank constant [eV s] (no 2 pi)
        h = 6.582119569e-16
        one = 1e-9  # GHz s
        beta = 1 / (temp * kb / h * one)  # inverse temperature [h/GHz]
        self.beta = beta
        beta_local = 1 / (temp_local * kb / h * one)
        self.beta_local = beta_local
        # construct initial density matrix
        eigvalue, eigvector = self.init_eigen(dtype)

        dE = eigvalue[:] - eigvalue[0]
        pr = np.exp(-beta * dE)
        pr = pr / sum(pr)
        if debug:
            print("dE", dE)
            print("pr", pr, "total", sum(pr))

        rho = np.zeros((eigvalue.size * eigvalue.size))
        for i in range(eigvalue.size):
            rho = rho + (pr[i]) * np.kron(eigvector[:, i], np.conj(eigvector[:, i]))
        rho = (1.0 + 0.0j) * rho
        return rho

    def _init_Fock(self) -> Tuple[ndarray, ndarray, ndarray]:
        r"""Computes pauli matrix tensor products

        Returns:
            ``sigma^x_i \otimes 1``,
            ``sigma^z_i \otimes 1``,
            ``sigma^z_i \otimes sigma^z_j \otimes 1``
        """
        FockX, FockZ, FockProj_0, Fockproj1, Fockplus, Fockminus = (
            [sp.csr_matrix(mmat) for mmat in mat]
            for mat in _init_Fock(self.graph["total_qubits"])
        )
        FockZZ = [[m1 @ m2 for m1 in FockZ] for m2 in FockZ]
        return FockX, FockZ, FockZZ, FockProj_0, Fockproj1, Fockplus, Fockminus

    def pushtoFock(self, i: int, local: ndarray) -> ndarray:
        """Tensor product of `local` at particle index i with 1 in fock space

        Arguments:
            i: particle index of matrix local
            local: matrix operator
        """
        fock = np.identity(1)
        for j in range(self.graph["total_qubits"]):
            if j == i:
                fock = np.kron(fock, local)
            else:
                fock = np.kron(fock, ID2)
        return fock

    def _constructIsingH(self, Jij: ndarray, hi: ndarray) -> ndarray:
        """Computes Hamiltonian (``J_ij is i < j``, i.e., upper diagonal"""
        IsingH = sp.csr_matrix(
            (2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"])
        )
        for i in range(self.graph["total_qubits"]):
            IsingH += hi[i] * self.FockZ[i]
            for j in range(i):
                IsingH += Jij[j, i] * self.FockZZ[i][j]
        return IsingH

    def _constructtransverseH(self, hxi: ndarray) -> ndarray:
        r"""Construct sum of tensor products of ``\sigma^x_i \otimes 1``
        """
        transverseH = sp.csr_matrix(
            (2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"]),
            dtype=complex,
        )
        for i in range(self.graph["total_qubits"]):
            transverseH += hxi[i] * self.FockX[i]
        return transverseH

    def _Bij(self, B: ndarray) -> ndarray:
        """J_ij coefficients for final Annealing Hamiltonian

        https://www.dwavesys.com/sites/default/files/
          14-1002A_B_tr_Boosting_integer_factorization_via_quantum_annealing_offsets.pdf

        Equation (1)

        Arguments:
            B: Anneal coefficients for given schedule.
        """
        return np.asarray(
            [
                [np.sqrt(B[i] * B[j]) for i in range(self.graph["total_qubits"])]
                for j in range(self.graph["total_qubits"])
            ]
        )

    def annealingH(self, s: float) -> ndarray:
        """Computes ``H(s) = A(s) H_init + B(s) H_final`` in units of "energyscale"
        """
        AxtransverseH = self._constructtransverseH(
            (self.AS.A(s) + self.offset["Aoffset"]) * np.ones(self.graph["total_qubits"])
        )
        BxIsingH = self._constructIsingH(
            self._Bij(self.AS.B(s)) * self.ising["Jij"], self.AS.B(s) * self.ising["hi"]
        )
        H = self.ising["energyscale"] * (-1 * AxtransverseH + BxIsingH)
        return H

    # @jit(nopython=True)
    def solve_pure(
            self, y1: ndarray, ngrid: int = 11, debug: bool = False
    ) -> PureSolutionInterface:
        """Solves time depepdent Schrödinger equation for pure inital state
        """
        start = self.offset["normalized_time"][0]
        end = self.offset["normalized_time"][1]
        interval = np.linspace(start, end, ngrid)

        sol = PureSolutionInterface(y1)

        for jj in range(ngrid - 1):
            y1 = y1 / (np.sqrt(np.absolute(y1.conj().T @ y1)))
            tempsol = solve_ivp(
                fun=self._apply_H,
                t_span=[interval[jj], interval[jj + 1]],
                y0=y1,
                t_eval=np.linspace(*self.offset["normalized_time"], num=100),
                **self.solver,
            )
            y1 = tempsol.y[:, tempsol.t.size - 1]
            sol.t = np.hstack((sol.t, tempsol.t))
            sol.y = np.hstack((sol.y, tempsol.y))

        if debug:
            print(
                "final total prob",
                (np.absolute(sol.y[:, -1].conj().dot(sol.y[:, -1]))) ** 2,
            )
        return sol

    def _annealingH_densitymatrix(self, s: float) -> ndarray:
        """Tensor product of commutator of annealing Hamiltonian with id in Fock space

        Code:
            H(s) otimes 1 - 1 otimes H(s)
        """
        Fockid = sp.eye(self.Focksize)
        return sp.kron(self.annealingH(s), Fockid) - sp.kron(Fockid, self.annealingH(s))

    def _apply_tdse_dense(self, t: float, y: ndarray) -> ndarray:
        """Computes ``-i [H(s), rho(s)]`` for density vector `y`"""
        f = -1j * self._annealingH_densitymatrix(t).dot(y)
        return f

    def _apply_tdse_dense2(self, t: float, y: ndarray) -> ndarray:
        """Computes ``-i [H(s), rho(s)]`` for density vector ``y`` by reshaping ``y``
        """
        # f = -1j * np.dot(self.annealingH_densitymatrix(t), y)
        # print('waht', type(self.Focksize))
        # print(self.Focksize)
        ymat = y.reshape((self.Focksize, self.Focksize))
        H = self.annealingH(t)
        if self.gamma == 0:
            lindblad = 0
        else:
            #gamma_t = np.mean(self.gamma*(self.AS.B(t)/self.AS.B(1)))
            #gamma_t = self.gamma*np.exp(-((t-1)/0.05)**2)
            gamma_t = self.gamma
            lindblad = self.get_lindblad(ymat, gamma_t, H, t) # full counting
        if self.gamma_local == 0:
            lindblad_local = 0
        else:
            #glocal_t = np.mean(self.gamma*(self.AS.B(t)/self.AS.B(1)))
            #glocal_t = np.mean(self.gamma_local*self.AS.A(t)/self.AS.A(0))
            glocal_t = self.gamma_local
            lindblad_local = self.get_lindblad2(ymat, glocal_t, H, t) # local decoherence
        #self.gammadict["s"].append(t)
        #self.gammadict["g"].append(gamma_t)
        #self.gammadict["glocal"].append(glocal_t)
        ymat = -1j * (H.dot(ymat) - ymat @ H)
        ymat += lindblad
        ymat += lindblad_local
        f = ymat.reshape(self.Focksize ** 2)
        return f

    def get_lindblad2(self,ymat,gamma,H, t):
        ''' gamma: decoherence rate = 1/(decoherence time), the unit is the same as the Hamiltonian
        '''
        lindblad = np.zeros(((self.Focksize, self.Focksize)), dtype=complex)
        for i in range(self.graph["total_qubits"]):
            gap = 2.0 * abs((self.ising["hi"])[i])
            e = np.exp(-self.beta_local * self.AS.B(t) * gap)
            if ((self.ising["hi"])[i] > 0):
                lindblad = lindblad + 2.0 * (self.Fockplus[i]) @ (ymat) @ (self.Fockminus[i]) - (self.Fockproj0[i]) @ (
                    ymat) - (ymat) @ (self.Fockproj0[i])
                lindblad = lindblad + e[i] * (
                            2.0 * (self.Fockminus[i]) @ (ymat) @ (self.Fockplus[i]) - (self.Fockproj1[i]) @ (ymat) - (
                        ymat) @ (self.Fockproj1[i]))
            else:
                lindblad = lindblad + 2.0 * (self.Fockminus[i]) @ (ymat) @ (self.Fockplus[i]) - (self.Fockproj1[i]) @ (
                    ymat) - (ymat) @ (self.Fockproj1[i])
                lindblad = lindblad + e[i] * (
                            2.0 * (self.Fockplus[i]) @ (ymat) @ (self.Fockminus[i]) - (self.Fockproj0[i]) @ (ymat) - (
                        ymat) @ (self.Fockproj0[i]))
        lindblad = gamma * lindblad
        return lindblad

    def get_lindblad1(self, ymat, gamma, H):
        ''' gamma: decoherence rate = 1/(decoherence time), the unit is the same as the Hamiltonian
        '''
        lindblad = np.zeros(((self.Focksize, self.Focksize)))
        for i in range(self.graph["total_qubits"]):
            if ((self.ising["hi"])[i] > 0):
                lindblad = lindblad + 2.0 * (self.Fockplus[i]) @ (ymat) @ (self.Fockminus[i]) - (self.Fockproj0[i]) @ (
                    ymat) - (ymat) @ (self.Fockproj0[i])
            else:
                lindblad = lindblad + 2.0 * (self.Fockminus[i]) @ (ymat) @ (self.Fockplus[i]) - (self.Fockproj1[i]) @ (
                    ymat) - (ymat) @ (self.Fockproj1[i])
        lindblad = gamma * lindblad
        return lindblad

    def get_lindblad3(self, ymat, gamma, H, t):
        '''full counting statistics under wide-band-limit
        '''
        value, vector = np.linalg.eigh(H.toarray())
        lindblad = np.zeros((len(value), len(value)), dtype=complex)
        for j in range(len(value)):
            for i in range(j):
                gap = value[j] - value[i]
                e = np.exp(-self.beta * gap / self.ising["energyscale"])
                # p=e/(1.0+e)
                lowering = np.kron(vector[:, i], np.conjugate(vector[:, j]))
                lowering = lowering.reshape(H.shape)
                raising = np.conjugate(np.transpose(lowering))

                # store some matrix multiplications to save computation
                rhoraising = (ymat) @ (raising)
                # if (e>1.e-10):
                rholowering = (ymat) @ (lowering)
                raisingrho = (raising) @ (ymat)
                # else:
                #    rholowering=np.zeros((len(value),len(value)),dtype=complex)
                #    raisingrho=np.zeros((len(value),len(value)),dtype=complex)
                loweringrho = (lowering) @ (ymat)

                # re-factored lindblad operator
                lindblad += ((lowering) @ (2.0 * rhoraising - e * (raisingrho)) + (raising) @ (
                            2.0 * e * (rholowering) - loweringrho) - (rhoraising) @ (lowering) - e * (rholowering) @ (
                                 raising))

        # lindblad=(1-p)*lower+p*np.conjugate(np.transpose(lower))
        # lindblad=(1-p)*( 2.0*(lower)@(ymat)@(raiseop)-raiseop@(lower)@(ymat)-(ymat)@(raiseop)@(lower) )
        # lindblad+=p*( 2.0*(raiseop)@(ymat)@(lower)-lower@(raiseop)@(ymat)-(ymat)@(lower)@(raiseop) )

        lindblad = gamma * lindblad
        return lindblad

    def get_lindblad(self,ymat,gamma,H,t):
        '''full counting statistics under wide-band-limit
        '''
        value, vector = np.linalg.eigh(H.toarray())
        lindblad = np.zeros((len(value), len(value)), dtype=complex)
        conjvector = np.conjugate(vector)

        # pre-compute
        rhoii = np.zeros(len(value), dtype=complex)
        proj = np.zeros((len(value), len(value), len(value)), dtype=complex)
        projrho = np.zeros((len(value), len(value), len(value)), dtype=complex)
        rhoproj = np.zeros((len(value), len(value), len(value)), dtype=complex)

        for i in range(len(value)):
            rhoii[i] = np.dot(conjvector[:, i], np.dot(ymat, vector[:, i]))
            proj[i, :, :] = np.kron(vector[:, i][None].T, conjvector[:, i])
            projrho[i, :, :] = np.kron(vector[:, i][None].T, np.dot(conjvector[:, i], ymat))
            rhoproj[i, :, :] = np.kron(np.dot(ymat, vector[:, i])[None].T, conjvector[:, i])

        for j in range(len(value)):
            for i in range(j):
                gap=value[j]-value[i]
                e=np.exp(-self.beta*gap/self.ising["energyscale"])

                lindblad += 2.0 * (rhoii[j] * proj[i, :, :] + e * rhoii[i] * proj[j, :, :]) - (
                            projrho[j, :, :] + rhoproj[j, :, :]) - e * (projrho[i, :, :] + rhoproj[i, :, :])

        lindblad = gamma * lindblad
        return lindblad

    def solve_mixed(self, rho: ndarray) -> ndarray:
        """Solves the TDSE
        """
        self.Focksize = int(np.sqrt(len(rho)))
        sol = solve_ivp(
            fun=self._apply_tdse_dense2,
            t_span=self.offset["normalized_time"],
            y0=rho,
            t_eval=np.linspace(*self.offset["normalized_time"], num=100),
            **self.solver,
        )
        return sol

    # Compute Correlations
    # One time correlation function
    def cZ(self, ti, xi, sol_densitymatrix):
        return np.trace(
            self.FockZ[xi]
            @ sol_densitymatrix.y[:, ti].reshape(
                2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"]
            ),
        )

    def c0(self, ti, xi, sol_densitymatrix):
        return np.trace(
            self.Fockproj0[xi]
            @ sol_densitymatrix.y[:, ti].reshape(
                2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"]
            ),
        )

    def c1(self, ti, xi, sol_densitymatrix):
        return np.trace(
            self.Fockproj1[xi]
            @ sol_densitymatrix.y[:, ti].reshape(
                2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"]
            ),
        )

    def cZZ(self, ti, xi, xj, sol_densitymatrix):
        return np.trace(
            self.FockZZ[xi][xj]
            @ sol_densitymatrix.y[:, ti].reshape(
                2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"]
            ),
        )

    def cZZd(self, ti, xi, xj, sol_densitymatrix):
        return self.cZZ(ti, xi, xj, sol_densitymatrix) - self.cZ(
            ti, xi, sol_densitymatrix
        ) * self.cZ(ti, xj, sol_densitymatrix)

    # Two time correlation function
    # http://qutip.org/docs/latest/guide/guide-correlation.html
    def cZZt2(self, ti, xi, sol_densitymatrixt2):
        return np.trace(
            self.FockZ[xi]
            @ sol_densitymatrixt2.y[:, ti].reshape(
                2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"]
            ),
        )

    def cZZt2d(self, ti, xi, tj, xj, sol_densitymatrix, sol_densitymatrixt2):
        return np.trace(
            self.FockZ[xi]
            @ sol_densitymatrixt2.y[:, ti].reshape(
                2 ** self.graph["total_qubits"], 2 ** self.graph["total_qubits"]
            ),
        ) - self.cZ(ti, xi, sol_densitymatrix) * self.cZ(tj, xj, sol_densitymatrix)

    # entanglement entropy

    def ent_entropy(self, rho, nA, indicesA, reg):
        """
        calculate the entanglement entropy
        input:
           rho: density matrix
           n: number of qubits
           nA: number of qubits in partition A
           indicesA: einsum string for partial trace
           reg: infinitesimal regularization
        """
        tensorrho = rho.reshape(
            tuple([2 for i in range(2 * self.graph["total_qubits"])])
        )
        rhoA = np.einsum(indicesA, tensorrho)
        matrhoA = rhoA.reshape(2 ** nA, 2 ** nA) + reg * np.identity(2 ** nA)
        s = -np.trace(matrhoA @ logm(matrhoA)) / np.log(2)
        return s

    def vonNeumann_entropy(self, rho, reg):
        totaln = self.graph["total_qubits"]
        matrho = rho.reshape(2 ** totaln, 2 ** totaln) + reg * np.identity(2 ** totaln)
        s = -np.trace(matrho @ logm(matrho)) / np.log(2)
        return s

    def q_mutual_info(self, rho, nA, nB, indicesA, indicesB, reg):
        """
        calculate the quantum mutual information
        """
        sa=ent_entropy(rho, nA, indicesA, reg)
        sb=ent_entropy(rho, nB, indicesB, reg)
        sab=vonNeumann_entropy(rho, reg)
        s=sa+sb-sab
        return s


    def find_partition(self) -> Tuple[int, str]:
        """
        Assumes that offset range is symmetric around zero.
        Splits partition to positive and negative offsets.
        Returns a np.einsum index string

        Example:
        nA = 2  # how many qubits in partition A
        indicesA = "ijmnijkl"  # einsum '1234 1234' qubits ie. if nA = 1 I trace out 3 out of 4 qubits
        """
        from string import ascii_lowercase as abc

        if self.offset["offset_min"] == 0:
            """
            If the offset if zero, partition same qubits as problems with offset.
            """
            op = dict(self.offset)
            op["offset_min"] = -0.1
            op["offset_range"] = 0.1
            A = AnnealSchedule(**op, graph_params=self.graph)
            offsets = A.offset_list
        else:
            offsets = self.AS.offset_list
        einidx1 = ""
        einidx2 = ""
        idx = 0
        nA = 0
        for offset in offsets:
            if offset < 0:
                einidx1 += abc[idx]
                einidx2 += abc[idx]
                idx += 1
            else:
                einidx1 += abc[idx]
                idx += 1
                einidx2 += abc[idx]
                idx += 1
                nA += 1
        einidx = einidx1 + einidx2
        ridx1 = ''
        ridx2 = ''
        for idx, eidx in enumerate(einidx1):
            if eidx != einidx2[idx]:
                ridx1 += eidx
                ridx2 += einidx2[idx]
            else:
                pass
        ridx = ridx1 + ridx2
        esum = f"{einidx}->{ridx}"
        print(esum)
        return nA, esum


@jit(nopython=True)
def _pushtoFock(i: int, local: ndarray, total_qubits: int) -> ndarray:
    """Tensor product of `local` at particle index i with 1 in fock space

    Arguments:
        i: particle index of matrix local
        local: matrix operator
    """
    fock = np.array([[1]], dtype=np.int8)
    for j in range(total_qubits):
        if j == i:
            fock = np.kron(fock, local)
        else:
            fock = np.kron(fock, ID2)
    return fock


_SIG_X = SIG_X.astype(np.int8)
_SIG_Z = SIG_Z.astype(np.int8)
_PROJ_0 = PROJ_0.astype(np.int8)
_PROJ_1 = PROJ_1.astype(np.int8)
_SIG_PLUS = SIG_PLUS.astype(np.int8)
_SIG_MINUS = SIG_MINUS.astype(np.int8)


@jit(nopython=True)
def _init_Fock(total_qubits: int) -> Tuple[ndarray, ndarray, ndarray]:
    r"""Computes pauli matrix tensor products

    Returns:
        ``sigma^x_i \otimes 1``,
        ``sigma^z_i \otimes 1``,
        ``sigma^z_i \otimes sigma^z_j \otimes 1``
    """
    shape = (total_qubits, 2 ** total_qubits, 2 ** total_qubits)
    FockX = np.empty(shape, dtype=np.int8)
    FockZ = np.empty(shape, dtype=np.int8)
    Fockproj0 = np.empty(shape, dtype=np.int8)
    Fockproj1 = np.empty(shape, dtype=np.int8)
    Fockplus = np.empty(shape, dtype=np.int8)
    Fockminus = np.empty(shape, dtype=np.int8)
    for i in range(total_qubits):
        FockX[i] = _pushtoFock(i, _SIG_X, total_qubits)
        FockZ[i] = _pushtoFock(i, _SIG_Z, total_qubits)
        Fockproj0[i] = _pushtoFock(i, _PROJ_0, total_qubits)
        Fockproj1[i] = _pushtoFock(i, _PROJ_1, total_qubits)
        Fockplus[i] = _pushtoFock(i, _SIG_PLUS, total_qubits)
        Fockminus[i] = _pushtoFock(i, _SIG_MINUS, total_qubits)
    return FockX, FockZ, Fockproj0, Fockproj1, Fockplus, Fockminus


"""
# CODE FOR KL DIVERGENCE
# copied from Jupyter to here

        # KL divergence
        from scipy import special
        from scipy.special import rel_entr

        # print('hello',rel_entr(np.zeros(2),np.zeros(2)))
        nt = 11
        tgrid = np.linspace(0, 1, nt)
        dimH = 2 ** n
        # print(n)
        KLdiv = np.zeros(nt)
        KLdiv2 = np.zeros(nt)
        KLdiv3 = np.zeros(nt)
        KLdiv4 = np.zeros(nt)
        for i in range(nt):
            energy, evec = np.linalg.eigh(tdse.annealingH(tgrid[i]))
            midn = int(dimH / 2)
            # print(midn)
            # the KL divergence between mid eigens state and nearby eigen state distribution
            p = np.absolute(np.conj(evec[:, midn]) * evec[:, midn])
            q = np.absolute(np.conj(evec[:, midn - 1]) * evec[:, midn - 1])
            KLdiv[i] = np.sum(rel_entr(p, q))
            KLdiv2[i] = np.sum(rel_entr(q, p))

            # the KL divergence between 1st and gnd state distribution
            p = np.absolute(np.conj(evec[:, 0]) * evec[:, 0])
            q = np.absolute(np.conj(evec[:, 1]) * evec[:, 1])
            KLdiv3[i] = np.sum(rel_entr(p, q))
            KLdiv4[i] = np.sum(rel_entr(q, p))

        plt.figure()
        plt.plot(tgrid, KLdiv)
        plt.plot(tgrid, KLdiv2)
        plt.plot(tgrid, KLdiv3)
        plt.plot(tgrid, KLdiv4)
        plt.legend(['mid/mid-1', 'mid-1/mid', 'gnd/1st', '1st/gnd'])
        plt.title('KL divergence')
        # end KL divergence

        # correlation functions
        xgrid = np.arange(n)
        # print(qubo.todense())

        plt.figure()
        data = np.asarray(
            [
                [
                    np.absolute(tdse.cZ(t, xgrid[x], sol_densitymatrix))
                    for t in range(sol_densitymatrix.t.size)
                ]
                for x in range(n)
            ]
        )

        # how correlated with sigma_z
        plt.figure("Z_i")
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        for idx, datai in enumerate(data):
            ax.errorbar(x=sol_densitymatrix.t, y=datai, label=f"qubit {idx}")
        ax.set_title(r"$|< Z_i >|$")
        ax.set_xlabel(r"$t_i$")
        ax.set_ylabel(r"$x_i$")
"""
"""
# PLOT CORRELATION FUNCTIONS
        plt.figure()
        data = np.asarray(
            [
                [
                    np.absolute(tdse.cZZ(t, xj, xgrid[x], sol_densitymatrix))
                    for t in range(sol_densitymatrix.t.size)
                ]
                for x in range(n)
            ]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos = ax.imshow(data)
        ax.set_aspect("auto")
        ax.set_title(r"$|< Z_i Z_j >|$")
        ax.set_xlabel(r"$t_i$")
        ax.set_ylabel(r"$x_i$")
        fig.colorbar(pos)

        plt.figure()
        data = np.asarray(
            [
                [
                    np.absolute(tdse.cZZd(t, xgrid[x], xj, sol_densitymatrix))
                    for t in range(sol_densitymatrix.t.size)
                ]
                for x in range(n)
            ]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos = ax.imshow(data)
        ax.set_aspect("auto")
        ax.set_title(r"$|< Z_i Z_j>-< Z_i >< Z_j >|$")
        ax.set_xlabel(r"$t_i$")
        ax.set_ylabel(r"$x_i$")
        fig.colorbar(pos)


        plt.figure()
        t = 0
        data = np.asarray(
            [
                [
                    np.absolute(tdse.cZZd(t, xgrid[xi], xgrid[xj], sol_densitymatrix))
                    for xi in range(n)
                ]
                for xj in range(n)
            ]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos = ax.imshow(data)
        ax.set_aspect("auto")
        ax.set_title(r"$|< Z_i Z_j>-< Z_i >< Z_j >|$, initial")
        ax.set_xlabel(r"$x_j$")
        ax.set_ylabel(r"$x_i$")
        fig.colorbar(pos)

        plt.figure()
        t = sol_densitymatrix.t.size - 1
        data = np.asarray(
            [
                [
                    np.absolute(tdse.cZZd(t, xgrid[xi], xgrid[xj], sol_densitymatrix))
                    for xi in range(n)
                ]
                for xj in range(n)
            ]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos = ax.imshow(data)
        ax.set_aspect("auto")
        ax.set_title(r"$|< Z_i Z_j>-< Z_i >< Z_j >|$, final")
        ax.set_xlabel(r"$x_j$")
        ax.set_ylabel(r"$x_i$")
        fig.colorbar(pos)


        # two time correlation functions
        # need to solve again using different initial density matrix...

        # choose one xj... if you want other xj you have to do this for each xj
        tj = 0
        xj = 0
        rho2 = np.dot(tdse.FockZ[xj], rho.reshape(2 ** n, 2 ** n))
        sol_densitymatrixt2 = tdse.solve_mixed(rho2.reshape(4 ** n))

        plt.figure()
        data = np.asarray(
            [
                [
                    np.absolute(tdse.cZZt2(t, xgrid[x], sol_densitymatrixt2))
                    for t in range(sol_densitymatrix.t.size)
                ]
                for x in range(n)
            ]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos = ax.imshow(data)
        ax.set_aspect("auto")
        ax.set_title(r"$ double time |< Z_i(t) Z_j>|$")
        ax.set_xlabel(r"$t_i$")
        ax.set_ylabel(r"$x_i$")
        fig.colorbar(pos)

        plt.figure()
        data = np.asarray(
            [
                [
                    np.absolute(
                        tdse.cZZt2d(
                            t, xgrid[x], tj, xj, sol_densitymatrix, sol_densitymatrixt2
                        )
                    )
                    for t in range(sol_densitymatrix.t.size)
                ]
                for x in range(n)
            ]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos = ax.imshow(data)
        ax.set_aspect("auto")
        ax.set_title(r"$ double time |< Z_i(t) Z_j>-< Z_i(t) >< Z_j >|$")
        ax.set_xlabel(r"$t_i$")
        ax.set_ylabel(r"$x_i$")
        fig.colorbar(pos)
"""
