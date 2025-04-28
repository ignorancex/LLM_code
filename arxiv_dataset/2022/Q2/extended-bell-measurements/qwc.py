import numpy as np
import scipy
from openfermion.ops import QubitOperator
from openfermion.measurements import group_into_tensor_product_basis_sets
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info.operators import Pauli
from xbm import get_vec_ab, expand_mat, get_nb_qubits, get_trace
from utils import mat2H, symbol_check


def mat2operator(mat=None, H=None):

    if H is None:
        H = mat2H(mat)

    operator = QubitOperator()
    for h in H:
        if h[1] != ['I0']:
            operator += QubitOperator(' '.join(h[1]), h[0])

    return operator


def get_meas_circuit_qwc(base_circ, symbols):

    symbols_org = symbols.copy()

    nb_qubits = len(list(symbols)[0])
    qregs = base_circ.qregs

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

    return circ


def get_all_meas_circs_bits_coefs_qwc(psi_0, psi_1, mat_A, check_symbol=True):

    nb_qubits = get_nb_qubits(mat_A)

    if psi_1 is not None:
        mat_A = expand_mat(mat_A)
        nb_qubits += 1

    if psi_1 is not None:
        base_circ = get_vec_ab(psi_0, psi_1)
    else:
        base_circ = psi_0

    H = mat2H(mat_A)
    operator = mat2operator(mat_A, H)
    cliques = group_into_tensor_product_basis_sets(operator)

    all_symbols = []
    for k, v in cliques.items():
        symbol_set = set()
        for vv in v.get_operators():
            symbol = '*' * nb_qubits
            for ps in list(vv.terms.keys())[0]:
                p, s = ps
                symbol = symbol[:p] + s + symbol[p+1:]
            symbol_set = symbol_set | set([symbol])
        all_symbols += [symbol_set]

    if get_trace(mat_A) != 0.:
        all_symbols += [{'*'*nb_qubits}]

    if check_symbol:
        all_symbols = symbol_check(H, all_symbols, nb_qubits, ignore_error=True)

    meas = []

    for symbols in all_symbols:

        circ = base_circ.copy()

        meas_circ = get_meas_circuit_qwc(circ, symbols)

        bitscoef = []
        for pauli_string in list(symbols):
            P = Pauli(label=''.join(list(pauli_string.replace('*', 'I')))).to_matrix()
            # TODO more efficiently
            coef = np.sum(mat_A * np.conjugate(P)) / 2**nb_qubits

            eigens = 1
            for s in pauli_string:
                if s == '*':
                    eigens = np.kron(eigens, [1, 1])
                else:
                    eigens = np.kron(eigens, [1, -1])

            for i in range(2**nb_qubits):
                bits = bin(i)[2:].zfill(nb_qubits)
                bitscoef += [(bits, eigens[i] * coef)]

        meas += [(circ.combine(meas_circ), bitscoef)]

    return meas
