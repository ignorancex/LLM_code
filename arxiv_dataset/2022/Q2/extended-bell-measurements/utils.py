import itertools
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from qiskit import transpile
from qiskit.quantum_info.operators import Pauli
from qiskit.providers.ibmq.managed import IBMQJobManager

from xbm import get_nb_qubits, get_rows_cols, get_matrix_component


def assign_params(circ, params_dict):

    params_assign = {}
    for pvec, param in params_dict.items():
        for pv, pr in zip(pvec.params, param):
            if pv in circ.parameters.data:
                params_assign.update({pv: pr})

    circ = circ.assign_parameters(params_assign)

    return circ


def eval_expectation(meas, qins, params_dict=None):

    exp = 0.

    all_circs = []
    for circ, bitscoeffs in meas:

        if circ is not None:

            if params_dict is None:
                assigned_circ = circ.copy()
            else:
                assigned_circ = assign_params(circ, params_dict).copy()

            if not qins.is_statevector:
                assigned_circ.measure_all()

            all_circs += [assigned_circ]

        else:

            exp += bitscoeffs[1]

    if qins.is_simulator:
        result = qins.execute(all_circs)
    else:
        job_manager = IBMQJobManager()
        transpiled_circs = transpile(all_circs, backend=qins.backend, seed_transpiler=qins.compile_config['seed_transpiler'])
        jobs = job_manager.run(transpiled_circs, backend=qins.backend, shots=qins.run_config.shots)

    if qins.is_statevector:

        job_idx = 0
        for circ, bitscoeffs in meas:

            if circ is not None:

                answer_state = result.get_statevector(job_idx)

                for bits, coef in bitscoeffs:
                    a = answer_state[int(bits, 2)]
                    exp += np.array(coef).astype(np.complex128) * a * np.conjugate(a)

                job_idx += 1

    else:

        if qins.is_simulator:
            count_list = result.get_counts()
            if type(count_list) != list:
                count_list = [count_list]
        else:
            managed_results = jobs.results()
            count_list = [managed_results.get_counts(i) for i in range(len(all_circs))]

        job_idx = 0
        for circ, bitscoeffs in meas:

            if circ is not None:

                answer_shots = {}
                for i in range(2**(circ.num_qubits)):
                    k = bin(i)[2:].zfill(circ.num_qubits)
                    counts = count_list[job_idx]
                    if k in counts.keys():
                        answer_shots.update({k: counts[k]})
                    else:
                        answer_shots.update({k: 0})

                for bits, coef in bitscoeffs:
                    a2 = answer_shots[bits] / sum(answer_shots.values())
                    exp += np.array(coef).astype(np.complex128) * a2

                job_idx += 1

    if qins.is_simulator:
        return exp, [result]
    else:
        results = [job.result() for job in jobs.jobs()]
        return exp, results


def eval_expectation_listed(meas_list, qins, params_dict=None):

    exp_list = [0.]*len(meas_list)
    all_circs_list = []

    for meas_idx, meas in enumerate(meas_list):

        all_circs = []
        for circ, bitscoeffs in meas:

            if circ is not None:

                if params_dict is None:
                    assigned_circ = circ.copy()
                else:
                    assign_params(circ, params_dict).copy()

                if not qins.is_statevector:
                    assigned_circ.measure_all()

                all_circs += [assigned_circ]

            else:

                exp_list[meas_idx] = exp_list[meas_idx] + bitscoeffs[1]

        all_circs_list += [all_circs]

    if qins.is_simulator:

        foo = []
        for all_circs in all_circs_list:
            foo += all_circs
        result = qins.execute(foo)

        if qins.is_statevector:

            job_idx = 0
            for meas_idx, meas in enumerate(meas_list):
                for circ, bitscoeffs in meas:

                    if circ is not None:

                        answer_state = result.get_statevector(job_idx)

                        for bits, coef in bitscoeffs:
                            a = answer_state[int(bits, 2)]
                            exp_list[meas_idx] = exp_list[meas_idx] + np.array(coef).astype(np.complex128) * a * np.conjugate(a)

                        job_idx += 1

        else:

            count_list = result.get_counts()
            if type(count_list) != list:
                count_list = [count_list]

            job_idx = 0
            for meas_idx, meas in enumerate(meas_list):
                for circ, bitscoeffs in meas:

                    if circ is not None:

                        answer_shots = {}
                        for i in range(2**(circ.num_qubits)):
                            k = bin(i)[2:].zfill(circ.num_qubits)
                            counts = count_list[job_idx]
                            if k in counts.keys():
                                answer_shots.update({k: counts[k]})
                            else:
                                answer_shots.update({k: 0})

                        for bits, coef in bitscoeffs:
                            a2 = answer_shots[bits] / sum(answer_shots.values())
                            exp_list[meas_idx] = exp_list[meas_idx] + np.array(coef).astype(np.complex128) * a2

                        job_idx += 1

        return exp_list, [[result]]

    else:

        job_manager = IBMQJobManager()
        foo = []
        for all_circs in all_circs_list:
            foo += all_circs
        transpiled_circs = transpile(foo, backend=qins.backend).copy()
        jobs = job_manager.run(transpiled_circs, backend=qins.backend, shots=qins.run_config.shots)
        managed_results = jobs.results()
        count_list = [managed_results.get_counts(i) for i in range(len(transpiled_circs))]

        job_idx = 0
        results_list = []
        for meas_idx, meas in enumerate(meas_list):
            for circ, bitscoeffs in meas:

                if circ is not None:

                    answer_shots = {}
                    for i in range(2**(circ.num_qubits)):
                        k = bin(i)[2:].zfill(circ.num_qubits)
                        counts = count_list[job_idx]
                        if k in counts.keys():
                            answer_shots.update({k: counts[k]})
                        else:
                            answer_shots.update({k: 0})

                    for bits, coef in bitscoeffs:
                        a2 = answer_shots[bits] / sum(answer_shots.values())
                        exp_list[meas_idx] = exp_list[meas_idx] + np.array(coef).astype(np.complex128) * a2

                    job_idx += 1

            results = [job.result() for job in jobs.jobs()]
            results_list += [results]

        return exp_list, results_list


def mat2H_naive(mat):

    pauliset = ['I', 'X', 'Y', 'Z']

    nb_qubits = int(np.log2(mat.shape[0]))
    H = []

    for v in itertools.product(pauliset, repeat=nb_qubits):
        strings = ''.join(list(v))
        P = Pauli(label=strings).to_matrix()
        coef = np.sum(mat * np.conjugate(P)) / 2**nb_qubits
        if coef != 0.:
            if strings.replace('I', '') != '':
                bar = []
                for i, s in enumerate(strings):
                    if s != 'I':
                        bar += [s+str(i)]
                H += [(coef, bar)]
            else:
                H += [(coef, ['I0'])]

    return H


def mat2H(mat_A):

    nb_qubits = get_nb_qubits(mat_A)
    rows, cols = get_rows_cols(mat_A)
    H = []
    pauli_dict = {'0': ['I', 'Z'], '1': ['X', 'Y']}

    if type(mat_A) == scipy.sparse.coo.coo_matrix:
        mat_A = mat_A.tolil()

    for rrcc in set(rows ^ cols):

        pauli_strings_list = []
        for rc in bin(rrcc)[2:].zfill(nb_qubits):

            pauli_strings_list += [pauli_dict[rc]]

        for pauli_string in itertools.product(*pauli_strings_list):

            sign = 1
            for s in pauli_string:
                if s in ['I', 'X']:
                    sign = np.kron(sign, [1, 1])
                elif s in ['Y']:
                    sign = np.kron(sign, [1j, -1j])  # Y conjugate
                elif s in ['Z']:
                    sign = np.kron(sign, [1, -1])

            coef = 0.
            for r in range(2**nb_qubits):
                c = (rrcc) ^ r
                coef += sign[r] * get_matrix_component(mat_A, r, c)

            coef /= 2.**nb_qubits

            reduced_string = []
            if set(pauli_string) != set('I'):
                for i, s in enumerate(pauli_string):
                    if s != 'I':
                        reduced_string += [s+str(i)]
                H += [(coef, reduced_string)]
            else:
                H += [(coef, ['I0'])]

    return H


def H2mat_naive(H, nb_qubits=None):

    if nb_qubits is None:
        nb_qubits = 0
        for value, symbols in H:
            for s in symbols:
                nb_qubits = max(nb_qubits, int(s[1:]))
        nb_qubits += 1

    mat = np.zeros((2**nb_qubits, 2**nb_qubits), np.complex128)

    for value, symbols in H:

        strings = 'I' * nb_qubits
        for s in symbols:
            pauli = s[0]
            idx = int(s[1:])
            strings = strings[:idx] + pauli + strings[idx+1:]

        mat += value * Pauli(label=strings).to_matrix()

    return mat


def symbols2pauli(symbols, nb_qubits):

    strings = 'I' * nb_qubits
    for s in symbols:
        pauli = s[0]
        idx = int(s[1:])
        strings = strings[:idx] + pauli + strings[idx+1:]

    return strings


def H2mat(H, nb_qubits=None, to_coo=False):

    if nb_qubits is None:
        nb_qubits = 0
        for value, symbols in H:
            for s in symbols:
                nb_qubits = max(nb_qubits, int(s[1:]))
        nb_qubits += 1

    if to_coo:
        rows = []
        cols = []
        data = []
    else:
        mat = np.zeros((2**nb_qubits, 2**nb_qubits), np.complex128)

    for value, symbols in H:

        pauli = symbols2pauli(symbols, nb_qubits)

        rc = ''
        sign = 1
        for s in pauli:
            if s == 'I':
                rc += '0'
                sign = np.kron(sign, [1, 1])
            elif s == 'X':
                rc += '1'
                sign = np.kron(sign, [1, 1])
            elif s == 'Y':
                rc += '1'
                sign = np.kron(sign, [-1j, 1j])
            elif s == 'Z':
                rc += '0'
                sign = np.kron(sign, [1, -1])
            else:
                raise ValueError(s)

        rc = int(rc, 2)

        for r in range(2**nb_qubits):
            c = rc ^ r
            if to_coo:
                rows += [r]
                cols += [c]
                data += [value * sign[r]]
            else:
                mat[r, c] += value * sign[r]

    if to_coo:
        mat = coo_matrix((data, (rows, cols)), shape=(2**(nb_qubits), 2**(nb_qubits)))

    return mat


def symbol_check(H, all_symbols, nb_qubits, ignore_error=False):

    symbol_set_1 = set()
    for h in H:
        ss = ''
        for s in h[1]:
            if s[0] == 'I':
                ss += '*'
            else:
                ss += s[0]
        if len(ss) < nb_qubits:
            ss += '*' * (nb_qubits-len(ss))
        symbol_set_1 = symbol_set_1 | set([ss])

    symbol_set_2 = set()
    for s in all_symbols:
        symbol_set_2 = symbol_set_2 | s

    if symbol_set_1 != symbol_set_2:
        if ignore_error:
            return all_symbols+[set([s]) for s in list(symbol_set_1-symbol_set_2)]
        else:
            raise ValueError('all_symbols does not include', symbol_set_1 - symbol_set_2)

    else:
        return all_symbols
