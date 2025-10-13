import sys
sys.path.append('..')
from quantum_geodesics import boilerplate, differential_equations
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
import numpy as np
from scipy.linalg import expm
from scipy.integrate import simpson

n_qubits = 2

U_target = Operator(QFT(n_qubits)).data

hamiltonian = boilerplate.complex_unitary_log(U_target)

phaseless_hamiltonian = hamiltonian - np.trace(hamiltonian)*np.eye(1<<n_qubits)/(1 << n_qubits)

q_vals, h_vals = solution = differential_equations.solve_eq88(
    phaseless_hamiltonian,
    n_qubits=n_qubits,
    n_points=11,
    matrix_ivp_points=11
)

print(f"{np.shape(q_vals) = }, {np.shape(h_vals) = }")

if False:

    U_target_exp = expm(-1j*phaseless_hamiltonian)

    time, h, u, f, g = differential_equations.solve_matrix_ivp(
        h_nought=phaseless_hamiltonian,
        n_qubits=n_qubits,
        penalty=1,
        t0=0,
        tf=1,
        n_points=101
    )

    a_matrix = boilerplate.cal_A(f, g, h)
    print(f"{a_matrix.shape = }")

    k_propagator = boilerplate.compute_propagator(
        a=a_matrix,
        dt=time[1] - time[0]
    )

    print(f"{k_propagator.shape = }")

    reshaped_k = np.reshape(
        k_propagator,
        (-1, 1 << n_qubits, 1 << n_qubits, 1 << n_qubits, 1 << n_qubits)
    )

    print(f"{reshaped_k.shape = }")

    u_dagger = boilerplate.dagger(u)

    j_propagator = simpson(
        y=np.einsum(
            'iad,idekl,ieb->iabkl',
            u_dagger,
            reshaped_k,
            u
        ),
        x=time,
        axis=0
    )

    print(f"{j_propagator.shape = }")


    j_mat = np.reshape(
        j_propagator,
        (-1, 1 << 2*n_qubits)
    )

    print(f"{j_mat.shape = }")

    j_mat_inv = boilerplate.numba_matrix_inverse(j_mat)


    print(f"{j_mat_inv.shape = }")

    reshaped_j_inv = np.reshape(
        j_mat_inv,
        (1 << n_qubits, 1 << n_qubits, 1 << n_qubits, 1 << n_qubits)
    )

    jacobi_arg = boilerplate.apply_super_operator(g, phaseless_hamiltonian)



