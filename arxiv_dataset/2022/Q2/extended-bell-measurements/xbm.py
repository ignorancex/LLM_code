import re
from typing import Union

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from qiskit import QuantumCircuit, QuantumRegister


def get_vec_ab(
    vec_a: QuantumCircuit,
    vec_b: QuantumCircuit,
    ancilla: Union[QuantumRegister, None] = None,
    label_a: str = 'controlled_a',
    label_b: str = 'controlled_b'
) -> QuantumCircuit:

    if ancilla is None:
        ancilla = QuantumRegister(1, name='ancilla')

    bits = []
    for qreg in vec_a.qregs:
        bits += qreg._bits
    bits += ancilla._bits

    vec_ab = QuantumCircuit(QuantumRegister(bits=bits))

    vec_ab.h(ancilla)

    controlled_a = vec_a.to_gate()
    controlled_a.label = label_a
    controlled_a = controlled_a.control()

    controlled_b = vec_b.to_gate()
    controlled_b.label = label_b
    controlled_b = controlled_b.control()

    bits_rev = ancilla._bits.copy()
    for qreg in vec_a.qregs:
        bits_rev += qreg._bits

    vec_ab.append(controlled_a, bits_rev)
    vec_ab.x(ancilla)
    vec_ab.append(controlled_b, bits_rev)

    return vec_ab


def get_exp_meas_circ(circ, T_rc, j0_rc, is_imag):
    
    meas_circ = circ.copy()

    meas_circ.h(j0_rc)

    for k in list(T_rc):
        if k != j0_rc:
            meas_circ.cx(j0_rc, k)

    if is_imag:
        meas_circ.s(j0_rc)

    return meas_circ.inverse()


def get_nb_qubits(mat):

    if scipy.sparse.issparse(mat):

        nb_qubits = int(np.log2(mat.get_shape()[0]))

    elif type(mat) == np.ndarray:

        nb_qubits = int(np.log2(mat.shape[0]))

    else:

        raise TypeError(type(mat))

    return nb_qubits


def expand_mat(mat):

    nb_qubits = get_nb_qubits(mat)

    if scipy.sparse.issparse(mat):

        mat = scipy.sparse.kron(np.array([[0, 2], [0, 0]]), mat)

    elif type(mat) == np.ndarray:

        mat = np.kron(np.array([[0, 2], [0, 0]]), mat)

    else:

        raise TypeError(type(mat))

    return mat


def get_rows_cols(mat):

    if scipy.sparse.issparse(mat):

        rows, cols = mat.nonzero()

    elif type(mat) == np.ndarray:

        rows, cols = np.where(mat != 0.)

    else:

        raise TypeError(type(mat))

    return rows, cols


def get_matrix_component(mat, row, col):

    if (scipy.sparse.isspmatrix_coo(mat) or
            scipy.sparse.isspmatrix_bsr(mat) or
            scipy.sparse.isspmatrix_dia(mat)):

        data = mat.getcol(col).getrow(row).data

        if len(data) == 0:
            A_rc = 0.
        else:
            A_rc = data[0]

    else:

        if type(mat) == np.ndarray or scipy.sparse.issparse(mat):

            A_rc = mat[row, col]

        else:

            raise TypeError(type(mat))

    return A_rc


def get_trace(mat):

    if scipy.sparse.issparse(mat):

        trace = mat.diagonal().sum()

    elif type(mat) == np.ndarray:

        trace = np.trace(mat)

    else:

        raise TypeError(type(mat))

    return trace


def get_all_meas_circs_bits_coefs(
    psi_0,
    psi_1,
    mat_A,
    part='both',
    reduce_nb_measure=True,
    return_circ=True
):

    nb_qubits = get_nb_qubits(mat_A)

    if psi_1 is not None:
        base_circ = get_vec_ab(psi_0, psi_1)
        mat_A = expand_mat(mat_A)
        nb_qubits += 1
    else:
        base_circ = psi_0

    rows, cols = get_rows_cols(mat_A)

    bits = []
    for qreg in base_circ.qregs:
        bits += qreg._bits

    meas_circ = QuantumCircuit(QuantumRegister(bits=bits))

    if type(mat_A) == scipy.sparse.coo.coo_matrix:
        mat_A = mat_A.tolil()

    dict_M = {}
    dict_G = {}

    for rr, cc in zip(rows, cols):

        A_rc = get_matrix_component(mat_A, rr, cc)

        if rr == cc:

            if 0 not in dict_M.keys():
                dict_M.update({0: (meas_circ.copy(), None)})
                dict_G.update({0: ({rr: A_rc}, None)})
            else:
                bitscoef_re, _ = dict_G[0]
                bitscoef_re.update({rr: A_rc})
                dict_G.update({0: (bitscoef_re, None)})

        else:

            if rr > cc:

                r = cc
                c = rr
                A_re = A_rc
                A_im = -A_rc

            else:

                r = rr
                c = cc
                A_re = A_im = A_rc

            ones = re.finditer('1', bin(r ^ c)[2:].zfill(nb_qubits)[::-1])
            zeros = re.finditer('0', bin(r)[2:].zfill(nb_qubits)[::-1])
            T_rc = set([m.span()[0] for m in ones])
            T0_r = set([m.span()[0] for m in zeros])

            j0_rc = np.array(list(T_rc & T0_r)).max()

            beta_rc_pls = r
            beta_rc_mns = r ^ (2**j0_rc)

            if r ^ c not in dict_M.keys():

                M_rc_Re = M_rc_Im = bitscoef_re = bitscoef_im = None

                if part == 'real' or part == 'both':
                    M_rc_Re = get_exp_meas_circ(meas_circ, T_rc, j0_rc, is_imag=False)
                if part == 'imag' or part == 'both':
                    M_rc_Im = get_exp_meas_circ(meas_circ, T_rc, j0_rc, is_imag=True)

                dict_M.update({r ^ c: (M_rc_Re, M_rc_Im)})
                if part == 'real' or part == 'both':
                    bitscoef_re = {beta_rc_pls: 0.5 * A_re, beta_rc_mns: -0.5 * A_re}
                if part == 'imag' or part == 'both':
                    bitscoef_im = {beta_rc_pls: 0.5j * A_im, beta_rc_mns: -0.5j * A_im}
                dict_G.update({r ^ c: (bitscoef_re, bitscoef_im)})

            else:

                bitscoef_re, bitscoef_im = dict_G[r ^ c]

                if part == 'real' or part == 'both':
                    bitscoef_re = dict_update(bitscoef_re, beta_rc_pls, 0.5 * A_re)
                    bitscoef_re = dict_update(bitscoef_re, beta_rc_mns, -0.5 * A_re)
                if part == 'imag' or part == 'both':
                    bitscoef_im = dict_update(bitscoef_im, beta_rc_pls, 0.5j * A_im)
                    bitscoef_im = dict_update(bitscoef_im, beta_rc_mns, -0.5j * A_im)

                dict_G.update({r ^ c: (bitscoef_re, bitscoef_im)})

    return dictMG2meas(base_circ, dict_M, dict_G, nb_qubits)


def dict_update(dic, k, v):
    if k in dic.keys():
        dic.update({k: dic[k]+v})
    else:
        dic.update({k: v})
    return dic


def dictMG2meas(base_circ, dict_M, dict_G, nb_qubits):

    keys = dict_M.keys()
    meas = []
    for key in keys:
        M_re, M_im = dict_M[key]
        bitscoef_re, bitscoef_im = dict_G[key]

        if bitscoef_re is not None:
            foo = []
            for k, v in bitscoef_re.items():
                foo += [(bin(k)[2:].zfill(nb_qubits), v)]
            meas += [(base_circ.combine(M_re), foo)]

        if bitscoef_im is not None:
            bar = []
            for k, v in bitscoef_im.items():
                bar += [(bin(k)[2:].zfill(nb_qubits), v)]
            meas += [(base_circ.combine(M_im), bar)]

    return meas
