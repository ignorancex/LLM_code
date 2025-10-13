from boilerplate import lie_bracket, PQ_decomp, cal_G, cal_F, cal_A, generate_pauli_bases, dagger, compute_propagator, complex_unitary_log, create_qft_matrix, get_F_G_matrices
import numpy as np
from scipy.integrate import solve_ivp, simpson
from typing import Tuple
from numba import njit, prange

@njit(parallel=True)
def get_bracket_of_PQ(
    h_values:np.ndarray,
) -> np.ndarray:
    
    bracketed_h = np.empty_like(h_values)
    for ii in prange(bracketed_h.shape[0]):

        bracketed_h[ii] = lie_bracket(*PQ_decomp(h_values[ii]))

    return bracketed_h

get_bracket_of_PQ(np.arange(2*4*4).reshape((2,4,4)).astype(np.complex128))

@njit
def hamiltonian_rhs(h, u, penalty=10):

    vanilla_rhs= -1j*cal_F(
        lie_bracket(
            h,
            cal_G(
                h,
                penalty
            )
        ), 
        penalty
    )
    
    dim=h.shape[1]
    
    #return vanilla_rhs-(1/dim)*np.trace(vanilla_rhs)*np.identity(dim)
    
    return vanilla_rhs

@njit
def u_rhs(h, u, penalty=10):

    return -1j*h@u

def hu_system(
    t:float,
    hu:np.ndarray,
    penalty:int=10,
    matrix_shape:tuple=(2,2),
    number_of_elements:int=4
) -> np.ndarray:
    
    h, u = hu[:number_of_elements], hu[number_of_elements:]
    h = h.reshape(matrix_shape, )
    u = u.reshape(matrix_shape, )

    h_dot = hamiltonian_rhs(h, u, penalty=penalty)
    u_dot = u_rhs(h, u, penalty=penalty)

    return np.concatenate((h_dot.flatten(), u_dot.flatten()))

def generate_random_u_target(
    n_qubits:int
) -> np.ndarray:
    
    u_target = np.zeros(
        (1 << n_qubits, )*2,
        dtype=np.complex128,
        
    )

    for _, pauli_basis in generate_pauli_bases(n_qubits=n_qubits):

        phase = np.random.uniform(1e-6, 2*np.pi)
        magnitude = np.random.uniform(1e-6, 10)
        # print((magnitude*np.exp(1j*phase)*pauli_basis).shape == u_target.shape)
        u_target += magnitude*np.exp(1j*phase)*pauli_basis

    return u_target

def get_u_nought(
    n_qubits:int
) -> np.ndarray:
    
    return np.eye(
        1 << n_qubits,
        dtype=np.complex128,
        
    )

def extract_hu(
    hu:np.ndarray,
    new_last_axes_shape:tuple
) -> Tuple[np.ndarray, np.ndarray]:
    
    hu = hu.T

    half_last_axis_length = hu.shape[-1]//2

    h, u = hu[:, :half_last_axis_length], hu[:, half_last_axis_length:]

    h = h.reshape((-1, *new_last_axes_shape), )
    u = u.reshape((-1, *new_last_axes_shape), )

    return h, u

def solve_matrix_ivp(
    h_nought:np.ndarray,
    n_qubits:int,
    penalty:float=10,
    t0:float=0,
    tf:float=1,
    n_points:int=101
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if h_nought.shape[0] != (1 << n_qubits):

        raise ValueError
    
    u_nought = get_u_nought(n_qubits=n_qubits)
    y0 = np.concatenate((h_nought.flatten(), u_nought.flatten()))
    result = solve_ivp(
        hu_system,
        t_span=(t0,tf),
        y0=y0,
        t_eval=np.linspace(t0, tf, n_points),
        args=(penalty, u_nought.shape, np.prod(u_nought.shape))
    )

    h_values, u_values = extract_hu(
        result.y,
        u_nought.shape
    )

    return result.t, h_values, u_values

def approximate_matrix_integral(
    time_points:np.ndarray,
    h_solution:np.ndarray,
    u_solution:np.ndarray
) -> np.ndarray:

    it_bracketed_h = np.complex128(1j)*time_points[:, np.newaxis, np.newaxis]*get_bracket_of_PQ(h_solution)
    
    u_solution_dagger = np.swapaxes(u_solution, -1, -2).conj()
    
    integrand = u_solution_dagger@it_bracketed_h@u_solution

    return simpson(integrand, x=time_points, axis=0)

if __name__ == '__main__':

    n_qubits = 3
    array_shape = (1 << n_qubits, )*2
    penalty = 10
    u_target = create_qft_matrix(n_qubits)
    h_nought = complex_unitary_log(u_target)

    t, h_values, u_values = solve_matrix_ivp(
        h_nought=h_nought,
        n_qubits=n_qubits,
        penalty=penalty,
        t0=0,
        tf=1,
        n_points=101
    )


    f, g = get_F_G_matrices(n_qubits, penalty)
    # print(h_values.shape)
    # print(g.shape)
    a = cal_A(f, g, h_values, penalty)
    k = compute_propagator(a)
    print(k.shape)
    print(h_values.shape, '\n', a.shape)
