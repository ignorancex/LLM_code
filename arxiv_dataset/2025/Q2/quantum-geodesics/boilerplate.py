import numpy as np
from numba import njit, objmode, prange
from numba.typed import Dict as numba_dict
from itertools import product, combinations, pairwise
from scipy.linalg import expm 
from typing import Generator, Tuple, Callable, List, Iterable, Dict
from timeit import Timer, timeit
from qiskit import QuantumCircuit
import math

I_2 = np.eye(
    2,
    dtype=np.complex128,
    order='C'
)
PAULI_X = np.array(
    [
        [0, 1],
        [1, 0]
    ],
    dtype=np.complex128,
    order='C'
)
PAULI_Y = np.array(
    [
        [0, -1j],
        [1j, 0]
    ],
    dtype=np.complex128,
    order='C'
)
PAULI_Z = np.array(
    [
        [1, 0],
        [0, -1]
    ],
    dtype=np.complex128,
    order='C'
)

def qft(n):
    """
    Applies the Quantum Fourier Transform (QFT) on the first n qubits of the given quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to apply the QFT to.
        n (int): The number of qubits to apply the QFT to.
    """
    circuit = QuantumCircuit(n)
    
    for qubit in range(n):
        # Apply Hadamard gate on the qubit
        circuit.h(qubit)
        
        # Apply controlled rotations on the following qubits
        for target_qubit in range(qubit + 1, n):
            angle = math.pi / float(2**(target_qubit - qubit))
            circuit.crz(angle, target_qubit, qubit)


    return circuit

@njit
def get_pauli_list() -> List[np.ndarray]: 
    return [
        np.eye(
            2,
            dtype=np.complex128,
        ),
        np.array(
            [
                [0, 1],
                [1, 0]
            ],
            dtype=np.complex128,
        ),
        np.array(
            [
                [0, -1j],
                [1j, 0]
            ],
            dtype=np.complex128,
        ),
        np.array(
            [
                [1, 0],
                [0, -1]
            ],
            dtype=np.complex128,
        )
    ]

@njit
def get_pauli_dict() -> Dict[str, np.ndarray]:

    return numba_dict(zip('ixyz', get_pauli_list()))

@njit
def lie_bracket(
    x:np.ndarray,
    y:np.ndarray,
    /
) -> np.ndarray:
    "Basic Lie Algebra commutator"

    return x@y - y@x

@njit
def dagger(
    x:np.ndarray       
) -> np.ndarray:

    return x.conj().T

@njit
def pauli_inner_product(
    a:np.ndarray,
    b:np.ndarray,
    /
) -> np.complex128:

    q = np.trace(a@dagger(b))

    return q/a.shape[0]

@njit
def trace_inner_product(
    a:np.ndarray,
    b:np.ndarray,
    /
) -> np.complex128:

    return np.trace(dagger(a)@b)

@njit
def n_choose_2(
    indexable:list | tuple | str,
    /
) -> Generator[tuple, None, None]:
    iter_length = len(indexable)
    for ii in range(iter_length):
        for jj in range(ii+1, iter_length):
            yield indexable[ii], indexable[jj]

tuple(n_choose_2(list(range(3))))

@njit 
def two_product(
    iterable:Iterable
) -> Generator[tuple, None, None]:
    
    for item1 in iterable:
        for item2 in iterable:
            yield item1, item2

list(two_product('xy'))

@njit
def n_product(
    indexable:str|list,
    repeat:int=1
):
    if repeat < 1:
        yield ''
    else:
        indexes = [0 for _ in range(repeat)]
        while True:
            yield ''.join([indexable[index] for index in indexes])
            j = len(indexes) - 1
            while True:
                indexes[j] += 1
                if indexes[j] < len(indexable): 
                    break
                indexes[j] = 0
                j -= 1
                if j < 0: 
                    return

@njit
def generate_P_signatures(n_qubits:int) -> Generator[str, None, None]:

    tuple_range = list(range(n_qubits))

    for ii in tuple_range:

        for character in 'xyz':

            yield ii*'i' + character + (n_qubits-ii-1)*'i'

    for (index1, index2) in n_choose_2(tuple_range):
        
        for (char1, char2) in two_product('xyz'):

            yield index1*'i' + char1 + (index2 - index1 - 1)*'i' + char2 + (n_qubits - index2 - 1)*'i'

list(generate_P_signatures(n_qubits=2))


@njit
def basis_constructor(
    pauli_operations:List[np.ndarray],
) -> np.ndarray:
    """
    Reduction operation making a Kron product on the list of Pauli operations
    left to right.
    """

    kron_product = np.array(
        [[1]], 
        dtype=np.complex128,
    )
    for pauli_matrix in pauli_operations:
        kron_product = np.kron(kron_product, pauli_matrix)
    return kron_product

basis_constructor(get_pauli_list())

@njit
def get_coeff_and_basis(
    a:np.ndarray,
    pauli_signature:tuple | str
) -> Tuple[complex, np.ndarray]:
    
    pauli_dict = get_pauli_dict()
    basis_matrix = basis_constructor([pauli_dict[key] for key in pauli_signature])
    basis_coeffi = pauli_inner_product(a, basis_matrix)

    return basis_coeffi, basis_matrix

@njit
def PQ_decomp(
    a:np.ndarray,
    /
) -> np.ndarray:
    
    num_rows = a.shape[0]
    n_qubits = 0
    while num_rows > 1:
        num_rows >>= 1
        n_qubits += 1
    
    p = np.zeros_like(a)

    for signature in generate_P_signatures(n_qubits=n_qubits):

        coeff, basis_matrix = get_coeff_and_basis(a, pauli_signature=signature)
        p += coeff*basis_matrix

    return p, a - p

@njit
def riemann_metric(
    x:np.ndarray,
    y:np.ndarray,
    penalty:float,
    /
) -> np.complex128:

    p, q = PQ_decomp(y)

    return (np.trace(x@p) + penalty*np.trace(x@q))/x.shape[0]

riemann_metric(I_2, I_2, 10)

@njit
def cal_F(
    x:np.ndarray,
    penalty:float,
    /
) -> np.ndarray:
    
    p, q = PQ_decomp(x)
    y = p + q/penalty

    return y

@njit
def cal_G(
    x:np.ndarray,
    penalty:float,
    /
) -> np.ndarray:
    
    p, q = PQ_decomp(x)
    y = p + penalty*q

    return y

@njit
def pauli_metric(
    x:np.ndarray,
    y:np.ndarray,
    penalty:float,
    /
) -> np.ndarray:
    
    return np.trace(x@cal_G(y, penalty))/x.shape[0]

pauli_metric(I_2, I_2, 10)

@njit
def cal_E(
    y:np.ndarray,
    x:np.ndarray,
) -> np.ndarray:

    return y - .5j*lie_bracket(x, y)

@njit    
def cal_D(
    y:np.ndarray,
    x:np.ndarray,
) -> np.ndarray:

    return y + .5j*lie_bracket(x, y)

cal_E(I_2, I_2)
cal_D(I_2, I_2)

@njit    
def cal_G_in_X(
    y:np.ndarray,
    x:np.ndarray,
    penalty:float,
) -> np.ndarray:
    
    ex = cal_E(y, x)
    Gex = cal_G(ex, penalty)
    extGex = cal_E(Gex, -x)
    return extGex

@njit    
def cal_F_in_X(
    y:np.ndarray,
    x:np.ndarray,
    penalty:float,
) -> np.ndarray:
    
    dxt = cal_D(y, -x)
    Fdxt = cal_F(dxt, penalty)
    dxFdxt = cal_D(Fdxt, x)
    return dxFdxt

@njit
def complex_unitary_log(
    U:np.ndarray
) -> np.ndarray:
    
    eigvals, eigvecs = np.linalg.eig(U)
    eigvecs_inv = np.linalg.inv(eigvecs)
    eigvals = np.exp(1j*np.angle(eigvals))  # np.angle shifts phase to (-pi, pi)
    log_diag = np.diag(np.log(eigvals))

    return 1j * eigvecs @ log_diag @ eigvecs_inv


def cal_A(
    f:np.ndarray,
    g:np.ndarray,
    H:np.ndarray,
    penalty:float,
) -> np.ndarray:
    """
    This function implements equation 73 in the paper. It expects a discretized H of shape (t, x, x), where t is the time axis.
    """
    I_N = np.eye(H.shape[1], dtype=np.complex128)
    A=np.empty((H.shape[0], H.shape[1]**2, H.shape[2]**2), dtype=np.complex128)

#    for t in list(range(H.shape[0])):
#        L = cal_G(H[t], penalty)
#        term1 = np.kron(L, I_N) - np.kron(I_N, L.T)
#        term2 = (np.kron(I_N, H[t].T) - np.kron(H[t], I_N)) @ g
#        A[t] = f @ (term1 + term2)
        
    for t in list(range(H.shape[0])):
        L = cal_G(H[t], penalty)
        #term1 = np.kron(I_N, L) - np.kron(L.T, I_N)
        term1=np.reshape(np.einsum('kb,lg->klbg',L,I_N)-np.einsum('kb,lg->klbg',I_N,L.T),(H.shape[1]**2,H.shape[1]**2))
        term2 = np.reshape((np.einsum('ke,la->klea',I_N,H[t].T)-np.einsum('ke,la->klea',H[t],I_N)),(H.shape[1]**2,H.shape[1]**2))@ g
        A[t] = f @ (term1 + term2)

    return A

@njit
def complex_unitary_log(
    U:np.ndarray
) -> np.ndarray:
    
    eigvals, eigvecs = np.linalg.eig(U)
    eigvecs_inv = np.linalg.inv(eigvecs)
    eigvals = np.exp(1j*np.angle(eigvals))  # np.angle shifts phase to (-pi, pi)
    log_diag = np.diag(np.log(eigvals))

    return 1j * eigvecs @ log_diag @ eigvecs_inv

def create_qft_matrix(
    n_qubits: int
) -> np.ndarray:
    
    N = 1 << n_qubits
    qft = np.fft.ifft(np.eye(N), norm="ortho")

    return qft


def metric_tensor_component(
    sigma:np.ndarray,
    tau:np.ndarray,
    basis:np.ndarray,
    penalty:float,
) -> complex | float:
    
    return np.trace(sigma@cal_G_in_X(tau, basis, penalty))/sigma.shape[0]

def inverse_tensor_component(
    sigma:np.ndarray,
    tau:np.ndarray,
    basis:np.ndarray,
    penalty:float,
) -> complex | float:
    
    return np.trace(sigma@cal_F_in_X(tau, basis, penalty))/sigma.shape[0]

@njit
def generate_pauli_signatures(n_qubits:int) -> Generator[Tuple[int, List[np.ndarray]], None, None]:

    signatures = n_product('ixyz', repeat=n_qubits)
    next(signatures)
    for signature in signatures:

        weight = n_qubits - signature.count('i')
        yield weight, signature

list(generate_pauli_signatures(3))

@njit
def generate_pauli_bases(n_qubits:int) -> Generator[Tuple[str, np.ndarray], None, None]:

    pauli_dict = get_pauli_dict()
    for weight, signature in generate_pauli_signatures(n_qubits=n_qubits):

        yield weight, basis_constructor([pauli_dict[key] for key in signature])

@njit
def is_power_of_2(number:int) -> bool:

    if number < 1:
        return False
    else:
        return (number & ~(number - 1)) == number

def generate_pauli_coeffs_and_bases(
    a:np.ndarray,
    /    
) -> Generator[Tuple[complex, np.ndarray], None, None]:
    
        if is_power_of_2(num_rows := a.shape[0]):

            n_qubits = 0
            while num_rows > 1:
                num_rows >>= 1
                n_qubits += 1
            
            for _, signature in generate_pauli_signatures(n_qubits=n_qubits):

                yield get_coeff_and_basis(a, pauli_signature=signature)

        else:
            raise ValueError()

@njit
def unweighted_christoffel(
    rho:np.ndarray,
    sigma:np.ndarray,
    tau:np.ndarray,
    penalty:float,
    \
) -> np.complex128:
     
    lie_bracket_section = lie_bracket(sigma, cal_G(tau, penalty)) + lie_bracket(tau, cal_G(sigma, penalty))
    f_weighting = cal_F(rho, penalty)

    return np.trace(f_weighting@lie_bracket_section)

unweighted_christoffel(np.eye(1, dtype=np.complex128),np.eye(1, dtype=np.complex128),np.eye(1, dtype=np.complex128),10)

def christoffel_symbols(
    pauli_bases:List[np.ndarray],
    penalty:float
) -> np.ndarray:

    basis_length = len(pauli_bases)

    symbols_array = np.empty(
        shape=(basis_length, basis_length*(basis_length-1) >> 1),
        dtype=pauli_bases[0].dtype
    )

    if basis_length < ((1 << 6) - 1):

        symbols_array[:] = 0
        return np.real(symbols_array)

    for ii, rho in enumerate(pauli_bases):

        for jj, (sigma, tau) in enumerate(combinations(pauli_bases, 2)):

            symbols_array[ii, jj] = unweighted_christoffel(
                rho,
                sigma,
                tau,
                penalty
            )

    return np.real(1j*symbols_array/(pauli_bases[0].shape[0] << 1))


def cumprod_mat(B):
    #take a list of matrices and do cumprod on them. Make sure to order them correctly
    cumprod_B=np.zeros(np.shape(B),dtype='complex128') #Make an array of zeros
    
    Nb=B.shape[0]
    
    for i,Bmat in enumerate(cumprod_B):#Fill them up with identities
        cumprod_B[i]=np.identity(B.shape[1],dtype='complex128')
    
    for i in range(Nb):
        for j in range(i):
            cumprod_B[i]=B[j]@cumprod_B[i]
    
    return cumprod_B
    
    
    
    

def compute_propagator(
    a:np.ndarray,
    dt:float
) -> np.ndarray:
    
    target_shape = (math.isqrt(a.shape[1]),)*4

    exp_a = expm(1j*a*dt)
    #flattened_K = np.cumprod(exp_a, axis=0)
    flattened_K = cumprod_mat(exp_a)
    #propagator = flattened_K.reshape(target_shape)

    return flattened_K

def check_lie_algebra_membership(
    a:np.ndarray,
    tolerance:float=1e-6
    \
) -> bool:
    
    return np.abs(np.trace(a)) < tolerance and np.allclose(dagger(a), a, rtol=tolerance, atol=tolerance)
 
@njit
def check_float_equality(
    x:complex|float|int,
    y:complex|float|int,
    /,
    relative_tolerance:float=1e-6,
    absolute_tolerance:float=1e-6
) -> bool:
    
    return abs(x - y) <= max(relative_tolerance*max(abs(x), abs(y)), absolute_tolerance)

@njit
def eigen_F_G_creator(
    n_qubits:int,
    penalty:float,
) -> Tuple[np.ndarray, np.ndarray]:
    
    reciprocal_penalty = 1/penalty

    G_eigen_value_list = np.ones(1 << 2*n_qubits, dtype=np.complex128)
    F_eigen_value_list = np.ones(1 << 2*n_qubits, dtype=np.complex128)
    eigen_vector_matrix = np.empty(
        shape=(1 << 2*n_qubits, 1 << 2*n_qubits),
        dtype=np.complex128
    )
    eigen_vector_matrix[0] = np.eye(
        1 << n_qubits, 
        dtype=np.complex128
    ).flatten()
    

    for index, (weight, basis) in enumerate(
        generate_pauli_bases(n_qubits=n_qubits),
        1
    ):
        if weight > 2:
            G_eigen_value_list[index] = penalty
            F_eigen_value_list[index] = reciprocal_penalty

        eigen_vector_matrix[index] = basis.flatten()

    G_eigen_value_matrix = np.diag(
        G_eigen_value_list
    ).astype(np.complex128)

    F_eigen_value_matrix = np.diag(
        F_eigen_value_list
    ).astype(np.complex128)

    eigen_vector_matrix = eigen_vector_matrix.T

    vector_matrix_inverse = np.linalg.inv(eigen_vector_matrix)
        
    G_mat = eigen_vector_matrix@G_eigen_value_matrix@vector_matrix_inverse
    F_mat = eigen_vector_matrix@F_eigen_value_matrix@vector_matrix_inverse

    return F_mat, G_mat

@njit
def eigen_P_Q_creator(
    n_qubits:int,
    \
) -> Tuple[np.ndarray, np.ndarray]:


    P_eigen_value_list = np.ones(1 << 2*n_qubits, dtype=np.complex128)
    Q_eigen_value_list = np.zeros(1 << 2*n_qubits, dtype=np.complex128)
    eigen_vector_matrix = np.empty(
        shape=(1 << 2*n_qubits, 1 << 2*n_qubits),
        dtype=np.complex128
    )
    eigen_vector_matrix[0] = np.eye(
        1 << n_qubits, 
        dtype=np.complex128
    ).flatten()

    for index, (weight, basis) in enumerate(
        generate_pauli_bases(n_qubits=n_qubits),
        1
    ):
        if weight > 2:
            P_eigen_value_list[index] = 0
            Q_eigen_value_list[index] = 1

        eigen_vector_matrix[index] = basis.flatten()

    P_eigen_value_matrix = np.diag(
        P_eigen_value_list
    ).astype(np.complex128)

    Q_eigen_value_matrix = np.diag(
        Q_eigen_value_list
    ).astype(np.complex128)

    eigen_vector_matrix = eigen_vector_matrix.T

    vector_matrix_inverse = np.linalg.inv(eigen_vector_matrix)
        
    P_mat = eigen_vector_matrix@P_eigen_value_matrix@vector_matrix_inverse
    Q_mat = eigen_vector_matrix@Q_eigen_value_matrix@vector_matrix_inverse

    return P_mat, Q_mat

@njit
def get_F_G_matrices(
    n_qubits:int,
    penalty:float,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if not penalty > 0:
        raise ValueError(f"Penalty parameter must be positive, but was penalty = {penalty}")

    reciprocal_penalty = 1/penalty

    identity_basis = np.eye(
        1 << n_qubits,
        dtype=np.complex128
    ).reshape((1 << 2*n_qubits, 1))

    F_mat = 0*identity_basis@identity_basis.T/(1 << n_qubits)
    G_mat = np.copy(F_mat)

    for weight, basis in generate_pauli_bases(n_qubits=n_qubits):

        flattened_basis = basis.reshape((basis.size, 1))
        outer_basis = flattened_basis@dagger(flattened_basis)/basis.shape[0]

        if weight > 2:
            F_mat += reciprocal_penalty*outer_basis
            G_mat += penalty*outer_basis

        else:
            F_mat += outer_basis
            G_mat += outer_basis

    return F_mat, G_mat

@njit
def get_P_Q_matrices(
    n_qubits:int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    identity_basis = np.eye(
        1 << n_qubits,
        dtype=np.complex128
    ).reshape((1 << 2*n_qubits, 1))

    P_mat = 0*identity_basis@identity_basis.T/(1 << n_qubits)
    Q_mat = np.zeros_like(P_mat)

    for weight, basis in generate_pauli_bases(n_qubits=n_qubits):

        flattened_basis = basis.reshape((basis.size, 1))
        outer_basis = flattened_basis@dagger(flattened_basis)/basis.shape[0]

        if weight > 2:
            Q_mat += outer_basis

        else:
            P_mat += outer_basis

    return P_mat, Q_mat


@njit
def apply_super_operator(
    super_op_matrix:np.ndarray,
    pauli_representable_matrix:np.ndarray
) -> np.ndarray:

    reshaped_matrix = np.reshape(pauli_representable_matrix, newshape=(super_op_matrix.shape[0], 1))
    return np.reshape(super_op_matrix@reshaped_matrix, newshape=pauli_representable_matrix.shape)

def compute_K_propagator(
    A_t:np.ndarray
) -> np.ndarray:
    
    exp_A = expm(A_t)
    flattened_K = np.prod(A_t)#, axis=)

    return flattened_K


if __name__ == '__main__':

    n_qubits = 2

    # P_mat, Q_mat = eigen_P_Q_creator(n_qubits)
    # P_mat, Q_mat = get_P_Q_matrices(n_qubits)
    # print(f"{np.allclose(P_mat, np.eye(P_mat.shape[0], dtype=P_mat.dtype)) = }")
    # print(f"{np.allclose(Q_mat, np.zeros_like(Q_mat)) = }")

    bases = list(item for _, item in generate_pauli_bases(n_qubits))

    # for i in range(10):

    #     coeffs = np.random.uniform(1e-6, 10, size=len(bases)).astype(np.complex128)
    #     phases = np.random.uniform(1e-6, 2*np.pi, size=len(bases))

    #     coeffs *= np.exp(1j*phases)

    #     test_mat = np.sum(list(map(lambda x: x[0]*x[-1], zip(coeffs, bases))), axis=0)

    #     old_one, old_two = PQ_decomp(test_mat)
    #     new_one = apply_super_operator(P_mat, test_mat)

    #     print(f"Test {i} P, {np.allclose(old_one, new_one, rtol=1e-10, atol=1e-10) = }")

    #     new_two = apply_super_operator(Q_mat, test_mat)

    #     print(f"Test {i} Q, {np.allclose(old_two, new_two, rtol=1e-10, atol=1e-10) = }")

    # penalty = 12

    # F_mat, G_mat = get_F_G_matrices(n_qubits, penalty)
    # print(f"{np.allclose(G_mat, np.eye(G_mat.shape[0], dtype=G_mat.dtype)) = }")
    # print(f"{np.allclose(F_mat, np.eye(F_mat.shape[0], dtype=F_mat.dtype)) = }")

    # for i in range(10):

    #     coeffs = np.random.uniform(1e-6, 10, size=len(bases)).astype(np.complex128)
    #     phases = np.random.uniform(1e-6, 2*np.pi, size=len(bases))

    #     coeffs *= np.exp(1j*phases)

    #     test_mat = np.sum(list(map(lambda x: x[0]*x[-1], zip(coeffs, bases))), axis=0)

    #     old_one = cal_F(test_mat, penalty)
    #     new_one = apply_super_operator(F_mat, test_mat)

    #     print(f"Test {i} F, {np.allclose(old_one, new_one, rtol=1e-10, atol=1e-10) = }")

    #     old_one = cal_G(test_mat, penalty)
    #     new_one = apply_super_operator(G_mat, test_mat)

    #     print(f"Test {i} G, {np.allclose(old_one, new_one, rtol=1e-10, atol=1e-10) = }")

else:
    
    get_pauli_dict()
    dagger(I_2)
    lie_bracket(I_2, I_2)
    pauli_inner_product(I_2, I_2)
    trace_inner_product(I_2, I_2)

CNOT_2 = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ],
    dtype=np.complex128,
)



# H = complex_unitary_log(create_qft_matrix(4))
# length = np.sqrt(np.trace(H @ H) / 2)
# assert (np.abs(length) - np.real(length)) < 1e-10, "check length calculation"
# print("complexity of operator: ", np.real(length))