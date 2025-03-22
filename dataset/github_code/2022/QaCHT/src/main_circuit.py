from qiskit import *
from qiskit import Aer
from qiskit.quantum_info import partial_trace
import numpy as np
from qiskit.circuit import Parameter
from itertools import combinations
import matplotlib.pyplot as plt
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def setpartition(iterable, n=2):
    iterable = list(iterable)
    partitions = combinations(combinations(iterable, n), r=int(len(iterable)/n))
    for partition in partitions:
        seen = set()
        for group in partition:
            if seen.intersection(group):
                break
            seen.update(group)
        else:
            yield partition


def setpartition_to_list(setpartition):
    """
    Input:
    ------
    setpartition: A permutation of causally connected qubits in form of sets.
    
    Output:
    A conversion to list of the set of permutations. 

    """
    qubit_partitions = []
    for el in list(setpartition):
        small_list = []
        for no in range(len(el)):
            small_list.append( list(el[no]) )
        qubit_partitions.append(small_list)
    return qubit_partitions

def circuit_subroutine(qc, control_qubits, target_qubits, bit):
    """
    Inputs:
    -------
    qc = Quantum circuit.
    control_qubits = list of control qubits.
    target_qubits = list of target qubits.
    bit = bit value corresponding to the permutations i.e. for 0: '00'
    
    Outputs:
    --------
    Aritra dar causality hypothesis circuit 

    """
    for target in target_qubits:
        total = control_qubits + target
        for i in range(len(bit)):
            if bit[i] == '1':
                qc.x([control_qubits[0] + i])
        qc.append( cbell(len(control_qubits)), total )
        for i in range(len(bit)):
            if bit[i] == '1':
                qc.x([control_qubits[0] + i])
    qc.barrier()
    return qc


def cbell(control_no):
    """
    Input:
    ------
    control_no = number of control qubits
    Output:
    -------
    Controlled Bell's Unitary.
    """
    qc_ent = QuantumCircuit(2)
    qc_ent.h([0])
    qc_ent.cx([0], [1])
    gate = qc_ent.to_gate(label = 'C-Bell Unitary').control(control_no)
    return gate

def bell():
    """
    Output:
    -------
    Bell's Unitary.
    """
    qc_ent = QuantumCircuit(2)
    qc_ent.h([0])
    qc_ent.cx([0], [1])
    gate = qc_ent.to_gate(label = 'Bell Unitary')
    return gate


def causal_oracle(theta_y):
    """
    Output:
    -------
    Causal Oracle.
    """
    qc_ent = QuantumCircuit(3)
    qc_ent.ry(theta_y, [1])
    qc_ent.cswap([2], [1], [0])
    
    gate = qc_ent.to_gate(label = 'C')
    return gate

def causal_oracle_iswap():
    """
    Output:
    -------
    Causal Oracle iSWAP(theta).
    """
    theta = Parameter('θ')
    circuit = QuantumCircuit(2)
    circuit.rxx(theta, 0, 1)
    circuit.ryy(theta, 0, 1)
    gate = circuit.to_gate(label = f'iSWAP').control(1)
    return gate

def aritra_dar_causality( aritra_dar_dimension , qubit_partitions, gate, theta_init, theta_oracle, theta_x ):
    """
    Input:
    ------
    aritra_dar_dimension = number of qubits in causal circuit (Bell connections),
    target_qubits = list of target qubits
    control_qubits = list of control qubits
    gate = 'rx', 'ry', 'rz', 'had'
    theta_init = angle of gates/ for 'had' only one angle (theta_init = 0)
    theta_oracle = angle of oracle ('cry')
    
    Output:
    -------
    Aritra dar causality hypothesis circuit
    """
    control_no = int( np.ceil( np.log2( aritra_dar_dimension ) ) )
    qc = QuantumCircuit( 2*aritra_dar_dimension + control_no+1 )

    target_qubits_total = qubit_partitions
    control_qubits = []
    for el in range(control_no):
        control_qubits.append( 2*aritra_dar_dimension + el )
        qc.h([2*aritra_dar_dimension+el])
    
    for q_rot in range(aritra_dar_dimension):
        if gate == 'rx':
            qc.rx( theta_init, q_rot )
        elif gate == 'ry':
            qc.ry( theta_init, q_rot )
        elif gate == 'rz':
            qc.rz( theta_init, q_rot )
        elif gate == 'had':
            qc.h(q_rot)

    num = 0
    t = target_qubits_total[0][0]
    for i in range(len(target_qubits_total)):
        if t == target_qubits_total[i][0]:
            bit = format(num, f'0{control_no}b')
            circuit_subroutine(qc, control_qubits, target_qubits_total[i], bit)
        else:
            t = target_qubits_total[i][0]
            num +=1
            bit = format(num, f'0{control_no}b')
            circuit_subroutine(qc, control_qubits, target_qubits_total[i], bit)
    
    # for _ in range(oracle_repeation):
    c_oracle = causal_oracle(theta_oracle)
    # print(c_oracle)
    qc.rx(theta_x, [2*aritra_dar_dimension + control_no])
    for causal_q in range(aritra_dar_dimension):
        qc.append( c_oracle, [causal_q, causal_q + aritra_dar_dimension, 2*aritra_dar_dimension + control_no])
    
    return qc

def aritra_dar_dosha( aritra_der_bortoni, ancilla ):
    """
    Input:
    ------
    aritra_der_bortoni = Is besically the causally connected circuit which is decomposed into
    A,B, and C subsystems. "C" subsystem is connect to subsystem "B" with 2-controlled Bell's Unitary
    and the subsystem "A" is connected to the subsystem "B" by Bell's Unitaries.

    Output:
    -------
    aritra_der_dosha = Is basically the statevector from the causally connected circuit.
    """
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(aritra_der_bortoni, simulator).result()
    statevector = result.get_statevector( aritra_der_bortoni )
    return partial_trace(statevector, [ancilla])

if __name__ == "__main__":
    
    hypothesis_list = ["identity", "swap-ry"]
    aritra_dar_dimension = 4
    total_qubit_required = 2*aritra_dar_dimension + int( np.ceil( np.log2( aritra_dar_dimension ) ) )
    partition = list(setpartition(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))
    qubit_partitions = setpartition_to_list(partition)
    
    gate = 'had'
    if gate == 'had':
        theta_init_list = [0.0]
        theta_init = theta_init_list[0]
    else:
        theta_init_list = np.arange(0, 2*np.pi, 0.5)
    
    hypothesis = hypothesis_list[1]
    
    if hypothesis == "identity":
        theta_oracle_list = [0.0]
        theta_x_list = [0.0]
    elif hypothesis == "swap-ry":
        theta_oracle_list = np.arange(0, 2*np.pi, np.pi/20)
        theta_x_list = np.arange(0, 2*np.pi, np.pi/20)

    for theta_x in theta_x_list:
        dict_prob = {}
        for theta_oracle in theta_oracle_list:
            theta_init, theta_oracle, theta_x = round(theta_init, 3), round(theta_oracle, 3), round(theta_x, 3)
            
            print(f'oracle angle {theta_oracle}, RX angle {theta_x}')
            print('--------------')
            aritra_dar_bortoni = aritra_dar_causality( aritra_dar_dimension , qubit_partitions, gate, theta_init, theta_oracle, theta_x )
            aritra_chiribella_dosha = aritra_dar_dosha(  aritra_dar_bortoni, total_qubit_required )
            # print(aritra_dar_bortoni.decompose())
            # print(aritra_dar_bortoni)
            # exit()
            # print(aritra_chiribella_dosha.data)
            aritra_chiribella_diaganalized = np.diag(aritra_chiribella_dosha.data)
            x = []
            for i in range(len(aritra_chiribella_diaganalized)):
                p = abs(aritra_chiribella_diaganalized[i])
                if p > 1e-5:
                    dict_prob[f'{bin(i)[2:].zfill( total_qubit_required )}'] = p**2
            file = f'data/dict_prob_initial_hypo_{hypothesis}_oracle_ang_{theta_oracle}_theta_x_{theta_x}_initial_initialization_{gate}.p'
            print(file)
            with open(file, 'wb') as handle:
                pickle.dump(dict_prob, handle)

# SWAP (Probability amplitude)
    # 00 0000 0000 0.5000000000000004
    # 01 0000 0000 0.25000000000000017
    # 01 0000 0101 0.25000000000000033
    # 01 0000 1010 0.25000000000000044
    # 01 0000 1111 0.25000000000000044
    # 10 0000 0000 0.25000000000000017
    # 10 0000 0110 0.2500000000000002
    # 10 0000 1001 0.25000000000000017
    # 10 0000 1111 0.2500000000000002
    # 11 0000 0000 0.25000000000000044
    # 11 0000 0011 0.2500000000000006
    # 11 0000 1100 0.25000000000000044
    # 11 0000 1111 0.25000000000000056
# IDENTITY
    # 00 0000 0000 0.5000000000000004
    # 01 0000 0000 0.25000000000000017
    # 01 0101 0000 0.25000000000000033
    # 01 1010 0000 0.25000000000000044
    # 01 1111 0000 0.25000000000000044
    # 10 0000 0000 0.25000000000000017
    # 10 0110 0000 0.2500000000000002
    # 10 1001 0000 0.25000000000000017
    # 10 1111 0000 0.2500000000000002
    # 11 0000 0000 0.25000000000000044
    # 11 0011 0000 0.2500000000000006
    # 11 1100 0000 0.25000000000000044
    # 11 1111 0000 0.25000000000000056

'''
    TODO------
    - measure only A (Z-basis)
    - check measure statistics 
    - compare between swap/identity
'''