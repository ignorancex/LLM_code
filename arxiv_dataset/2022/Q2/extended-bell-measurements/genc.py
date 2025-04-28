import numpy as np
import sympy
import scipy
from qiskit import QuantumCircuit, Aer
from qiskit.aqua import QuantumInstance
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info.operators import Pauli
from utils import mat2H, H2mat, symbol_check
import random
from xbm import get_vec_ab, get_nb_qubits, expand_mat, get_trace

'''
Download https://github.com/teaguetomesh/vqe-term-grouping/term_grouping.py
Rewrite line 257
-   return max_cliques
+   return max_cliques, term_reqs
'''
from term_grouping import genMeasureCircuit


def get_all_meas_circs_bits_coefs_gc(
    psi_0,
    psi_1,
    mat_A,
    commutativity_type,
    verbose=False,
    return_circ=True,
    check_symbol=True
):

    nb_qubits = get_nb_qubits(mat_A)

    if psi_1 is not None:
        base_circ = get_vec_ab(psi_0, psi_1)
        mat_A = expand_mat(mat_A)
        nb_qubits += 1
    else:
        base_circ = psi_0

    quantum_instance_unitary = QuantumInstance(Aer.get_backend('unitary_simulator'))

    H = mat2H(mat_A)

    all_symbols, term_reqs = genMeasureCircuit(H, nb_qubits, commutativity_type)

    if get_trace(mat_A) != 0.:
        all_symbols += [{'*'*nb_qubits}]

    if check_symbol:
        all_symbols = symbol_check(H, all_symbols, nb_qubits, ignore_error=True)

    comm_graph = commutativity_type.gen_comm_graph(term_reqs)

    if return_circ:

        mat_I = np.eye(2**nb_qubits)
        mat_0 = np.zeros((2**nb_qubits, 2**nb_qubits))
        success = False
        while not success:
            success = True
            meas = []

            for symbols in all_symbols:

                circ = base_circ.copy()

                meas_circ = get_meas_circuit(circ, symbols, comm_graph, term_reqs, verbose=verbose)

                unitary = quantum_instance_unitary.execute(meas_circ).get_unitary()

                bitscoef = []
                for pauli_string in list(symbols):

                    P = Pauli(label=''.join(list(pauli_string.replace('*', 'I')))).to_matrix()
                    # TODO more efficiently
                    coef = np.sum(mat_A * np.conjugate(P)) / 2**nb_qubits

                    mat_d = unitary @ P @ np.conjugate(unitary).T

                    if (all(np.ravel(np.isclose(np.abs(np.real(mat_d)), mat_I))) and
                            all(np.ravel(np.isclose(np.imag(mat_d), mat_0)))):
                        pass
                    else:
                        success = False
                        break

                    eigens = np.diagonal(mat_d)

                    for i in range(2**nb_qubits):
                        bits = bin(i)[2:].zfill(nb_qubits)
                        bitscoef += [(bits, eigens[i] * coef)]

                if not success:
                    print('Fail. Retry.')
                    break

                meas += [(circ.combine(meas_circ), bitscoef)]

    else:

        meas = all_symbols

    return meas


def symbols2stabmat(symbols, nb_qubits):

    stabilizer_matrix = np.zeros(shape=(len(symbols), 2*nb_qubits), dtype=np.int32)

    for i, pauli_string in enumerate(symbols):
        for j, p in enumerate(pauli_string):
            if p == 'X':
                stabilizer_matrix[i, j] = 1
            if p == 'Y':
                stabilizer_matrix[i, j] = 1
                stabilizer_matrix[i, nb_qubits+j] = 1
            if p == 'Z':
                stabilizer_matrix[i, nb_qubits+j] = 1

    return stabilizer_matrix


def mod2_gaussian_elimination(mat, row_start):

    row_start_org = row_start

    for col in range(mat.shape[1]):

        if mat[row_start, col] != 1:
            for row in range(row_start+1, mat.shape[0]):
                if mat[row, col] == 1:
                    mat[row_start] = (mat[row_start] + mat[row]) % 2
                    break

        if mat[row_start, col] == 1:
            for row in range(row_start+1, mat.shape[0]):
                if mat[row, col] == 1:
                    mat[row] = (mat[row] + mat[row_start]) % 2

            row_start += 1
            if row_start >= mat.shape[0]:
                break

    for row in range(row_start_org, mat.shape[0]):
        for col in range(row+1, mat.shape[1]):
            if mat[row, col] == 1:
                for tgt_row in range(row+1, mat.shape[0]):
                    if all(mat[tgt_row, :col] == 0) and mat[tgt_row, col] == 1:
                        mat[row] = (mat[row] + mat[tgt_row]) % 2
                        break

    return mat


def apply_swap(indices, circ):

    leng = len(indices)
    base = np.arange(leng)
    for i in range(leng):
        if base[i] != indices[i]:
            for j in range(leng):
                if indices[i] == base[j]:
                    foo = base[i]
                    base[i] = base[j]
                    base[j] = foo
                    circ.swap(i, j)
                    break

    return circ


def get_meas_circuit(base_circ, symbols, comm_graph, term_reqs, verbose=False):

    symbols_org = symbols.copy()

    nb_qubits = len(list(symbols)[0])
    qregs = base_circ.qregs

    if False:
        if len(symbols) < nb_qubits and len(symbols) != 1:
            i = 0
            while True:
                for j in range(nb_qubits):
                    dummy = list(symbols)[i]
                    dummy = dummy[:j]+'*'+dummy[j+1:]
                    symbols = symbols | {dummy}
                    if len(symbols) == nb_qubits:
                        break

                if len(symbols) == nb_qubits:
                    break

                i += 1

    flag = True
    symbol_list = list(symbols)

    if len(symbol_list) == 1:

        symbol_for_circ = symbol_list[0]

    else:

        symbol_for_circ = symbol_list[0]
        for i in range(nb_qubits):
            if symbol_for_circ[i] == '*':
                for symbol in symbol_list[1:]:
                    if symbol[i] != '*':
                        symbol_for_circ = symbol_for_circ[:i] + symbol[i] + symbol_for_circ[i+1:]

        for symbol in symbol_list:
            for i in range(nb_qubits):
                if (symbol[i] != symbol_for_circ[i] and
                        symbol[i] != '*' and
                        symbol_for_circ[i] != '*'):
                    flag = False

    if flag:

        if type(qregs) == list:
            circ = QuantumCircuit(*qregs)
        else:
            circ = QuantumCircuit(qregs)

        for i, s in enumerate(reversed(symbol_for_circ)):
            if s == 'X':
                circ.h(i)
            if s == 'Y':
                circ.sdg(i)
                circ.h(i)

    else:

        bits = []
        for qreg in qregs:
            bits += qreg._bits

        reg = QuantumRegister(bits=bits)
        circ = QuantumCircuit(reg)

        comm = set([''.join(s) for s in term_reqs])
        for symbol in list(symbols):
            comm = comm & set(comm_graph[symbol])
        comm = comm - symbols - set(['*'*nb_qubits])
        comm_list = list(comm)
        random.shuffle(comm_list)

        while True:
            '''
            Apply Gaussian elimination to the stabilizer matrix to eliminate
            the linearly dependend Pauli strings. If (number of linearly
            dependend Pauli strings) < (number of qubits), add commutative
            Pauli strings and recalculate the stabilizer matrix until (number
            of linearly dependend Pauli strings) = (number of qubits).
            '''
            success = True
            icomm = 0

            stabmat = symbols2stabmat(symbols, nb_qubits)  # [X, Z]
            stab_org = stabmat.copy()

            stabmat = mod2_gaussian_elimination(stabmat, 0)

            for row in range(len(symbols)):
                if all(stabmat[row] == 0):
                    stabmat = stabmat[:row]
                    break

            if len(stabmat) != nb_qubits:

                while True:

                    add_symbol = [comm_list[icomm]]
                    symbols = symbols | set(add_symbol)
                    stabmat = symbols2stabmat(symbols, nb_qubits)
                    stabmat = mod2_gaussian_elimination(stabmat, 0)

                    for row in range(len(symbols)):
                        if all(stabmat[row] == 0):
                            stabmat = stabmat[:row]
                            break

                    if len(stabmat) == nb_qubits:
                        break_flag = True
                        '''
                        Add Pauli strings so that stabilizer matrix does not
                        contain an all-zeros column. If there is all-zeros
                        column in stabilizer matrix, it will be unranked.
                        '''
                        for col in range(nb_qubits):
                            if all(stabmat[:, col] == 0):
                                break_flag = False
                                symbols -= set(add_symbol)
                                break
                    else:
                        break_flag = False

                    if break_flag:
                        break

                    icomm += 1

            stabmat_reduced = stabmat.copy()

            x_mat = stabmat[:, :nb_qubits]
            rank_x = np.linalg.matrix_rank(x_mat.astype(np.float32))
            rank_x_reduced = rank_x

            if rank_x != nb_qubits:

                rank_A = np.linalg.matrix_rank(stabmat[:rank_x, :nb_qubits])

                '''
                Find the full-rank submatrix C2 in matrix C and move it to
                the right, while simultaneously shifting the X matrix.
                A B      A1 A2 B1 B2
                0 C  ->  0  0  C1 C2
                '''
                mat_C = stabmat[rank_x:nb_qubits, nb_qubits:]
                _, inds = sympy.Matrix(mat_C).rref()
                inds = list(inds)
                remains = list(set(np.arange(mat_C.shape[1]))-set(inds))
                sorted_indices = remains + inds
                circ = apply_swap(sorted_indices.copy(), circ)
                sorted_indices += (np.array(sorted_indices) + nb_qubits).tolist()
                stabmat = stabmat[:, sorted_indices]

                stabmat_00C1C2 = stabmat.copy()
                A1 = stabmat[:rank_x, :rank_x]
                rank_A1 = np.linalg.matrix_rank(A1.astype(np.float32))

            '''
            Apply the Gaussian elimination method to the lower (n-k) rows.
            A1 A2 B1 B2      A1 A2 B1 B2
            0  0  C1 C2  ->   0  0  D  I
            '''
            stabmat = mod2_gaussian_elimination(stabmat, nb_qubits-rank_x)
            stabmat_00DI = stabmat.copy()

            '''
            Make the X matrix full rank by applying H gate
            on (n-k) qubits on the right side
            A1 A2 B1 B2      A1 B2 B1 A2
             0  0  D  I  ->   0  I  D  0
            '''
            indices = np.arange(rank_x).tolist()
            indices += (np.arange(nb_qubits-rank_x)+nb_qubits+rank_x).tolist()
            indices += (np.arange(rank_x)+nb_qubits).tolist()
            indices += (np.arange(nb_qubits-rank_x)+rank_x).tolist()
            stabmat = stabmat[:, indices]
            for i in range(rank_x, nb_qubits):
                circ.h(i)

            stabmat_0ID0 = stabmat.copy()

            rank_x_new = np.linalg.matrix_rank(stabmat[:, :nb_qubits].astype(np.float32))

            '''
            A1 B2 B1 A2
             0  I  D  0  ->  [I sym.]
            '''

            while True:

                '''
                First, make the X matrix a lower triangular matrix
                while setting its diagonal component to 1.
                '''
                for i in range(nb_qubits):
                    if stabmat[i, i] != 1:
                        for col in range(i+1, nb_qubits):
                            if stabmat[i, col] == 1:
                                foo = stabmat[:, col].copy()
                                stabmat[:, col] = stabmat[:, i]
                                stabmat[:, i] = foo
                                foo = stabmat[:, col+nb_qubits].copy()
                                stabmat[:, col+nb_qubits] = stabmat[:, i+nb_qubits]
                                stabmat[:, i+nb_qubits] = foo
                                circ.swap(i, col)
                                break

                    for col in range(i+1, nb_qubits):
                        if stabmat[i, col] == 1:
                            stabmat[:, col] = (stabmat[:, col] + stabmat[:, i]) % 2
                            stabmat[:, i+nb_qubits] = (stabmat[:, i+nb_qubits] + stabmat[:, col+nb_qubits]) % 2
                            circ.cx(i, col)
                '''
                Make the X matrix identity.
                '''
                for col in range(nb_qubits):
                    for row in range(col+1, nb_qubits):
                        if stabmat[row, col] == 1:
                            stabmat[:, col] = (stabmat[:, col] + stabmat[:, row]) % 2
                            stabmat[:, row+nb_qubits] = (stabmat[:, row+nb_qubits] + stabmat[:, col+nb_qubits]) % 2
                            circ.cx(row, col)

            stabmat_Isym = stabmat.copy()

            '''
            [I sym.]  ->  [I 0]
            '''
            for i in range(nb_qubits):
                if stabmat[i, i+nb_qubits] == 1:
                    stabmat[i, i+nb_qubits] = 0
                    circ.s(i)

            for i in range(nb_qubits):
                for j in range(i+1, nb_qubits):
                    if stabmat[i, j+nb_qubits] == 1:
                        stabmat[i, j+nb_qubits] = 0
                        stabmat[j, i+nb_qubits] = 0
                        circ.cz(i, j)

            stabmat_I0 = stabmat.copy()

            '''
            [I 0] -> [0 I]
            '''
            for i in range(nb_qubits):
                foo = stabmat[:, i].copy()
                stabmat[:, i] = stabmat[:, i+nb_qubits]
                stabmat[:, i+nb_qubits] = foo
                circ.h(i)

            stab_final = stabmat.copy()

            if (any(np.ravel(stabmat[:, :nb_qubits] != np.zeros((nb_qubits, nb_qubits)))) or
                    any(np.ravel(stabmat[:, nb_qubits:] != np.eye(nb_qubits)))):
                success = False

            break

        circ = circ.reverse_bits()

    return circ
