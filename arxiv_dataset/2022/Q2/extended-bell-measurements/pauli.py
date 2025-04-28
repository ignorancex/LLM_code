import numpy as np
import scipy
from qiskit.quantum_info.operators import Pauli
from qiskit import QuantumCircuit
import itertools
from xbm import get_vec_ab, expand_mat, get_nb_qubits, get_rows_cols


def get_bitscoef(pauli_string, coef):

    nb_qubits = len(pauli_string)

    bitscoef = []

    pp = ''.join(pauli_string)
    for i in range(2**nb_qubits):

        bits = bin(i)[2:].zfill(nb_qubits)

        nb_xyz = 0

        for p, b in zip(pp, bits):

            if p != 'I' and b == '1':
                nb_xyz += 1

        if not nb_xyz % 2:
            bitscoef += [(bits, coef)]
        else:
            bitscoef += [(bits, -coef)]

    return bitscoef


def get_meas_circ_pauli(qregs, pauli_string):

    if type(qregs) == list:
        meas_circ_pauli = QuantumCircuit(*qregs)
        qregs = [qq for q in qregs for qq in q]
    else:
        meas_circ_pauli = QuantumCircuit(qregs)

    for i, p in enumerate(reversed(pauli_string)):

        if p == 'X':

            meas_circ_pauli.h(qregs[i])

        elif p == 'Y':

            meas_circ_pauli.sdg(qregs[i])
            meas_circ_pauli.h(qregs[i])

    return meas_circ_pauli


def get_all_meas_circs_bits_coefs_pauli(psi_0, psi_1, mat_A, reduce=True):

    if psi_1 is not None:
        base_circ = get_vec_ab(psi_0, psi_1)
        mat_A = expand_mat(mat_A)
    else:
        base_circ = psi_0

    nb_qubits = get_nb_qubits(mat_A)

    rows, cols = get_rows_cols(mat_A)

    pauli_dict = {'0': ['I', 'Z'], '1': ['X', 'Y']}
    all_pauli_strings = []
    meas_dict = {}

    for rrcc in set(rows ^ cols):

        pauli_strings_list = []

        for rc in bin(rrcc)[2:].zfill(nb_qubits):

            pauli_strings_list += [pauli_dict[rc]]

        for pauli_string in itertools.product(*pauli_strings_list):
            all_pauli_strings += [pauli_string]

    for pauli_string in all_pauli_strings:

        P = Pauli(label=''.join(list(pauli_string))).to_matrix()

        # TODO more efficiently
        coef = np.sum(mat_A * np.conjugate(P)) / 2**nb_qubits

        if coef != 0.:

            bitscoef = get_bitscoef(pauli_string, coef)
            pp = ''.join(pauli_string)

            circ_key = pp.replace('Z', 'I')

            if reduce and (circ_key in meas_dict.keys()):

                circ, bitscoef_exist = meas_dict[circ_key]

                bitscoef_new = []
                for bc1, bc2 in zip(bitscoef, bitscoef_exist):
                    bits1, coef1 = bc1
                    bits2, coef2 = bc2
                    bitscoef_new += [(bits1, coef1+coef2)]

                meas_dict.update({circ_key: (circ, bitscoef_new)})

            else:

                exp_meas_circ = get_meas_circ_pauli(base_circ.qregs, pauli_string)

                circ = base_circ.combine(exp_meas_circ)

                if reduce:

                    meas_dict.update({pp.replace('Z', 'I'): (circ, bitscoef)})

                else:

                    meas_dict.update({pp: (circ, bitscoef)})

    return [v for v in meas_dict.values()]
