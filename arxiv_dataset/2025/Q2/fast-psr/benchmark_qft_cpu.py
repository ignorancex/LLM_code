def qft_Qiskit(num_qubits):
    import qiskit
    import numpy as np
    """QFT on the first n qubits in circuit"""
    def qft_rotations_Qiskit(qc: qiskit.QuantumCircuit, num_qubits):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if num_qubits == 0:
            return qc
        num_qubits -= 1
        qc.h(num_qubits)
        for j in range(num_qubits):
            
            qc.rz(np.pi/2**(num_qubits-j) / 2, num_qubits)
            qc.cx(j, num_qubits)
            qc.rz(-np.pi/2**(num_qubits-j) / 2, num_qubits)
            qc.cx(j, num_qubits)
            qc.rz(+np.pi/2**(num_qubits-j) / 2, num_qubits)
            
            # qc.cp(np.pi/2**(num_qubits-j), j, num_qubits)
            # qc.barrier()
        qft_rotations_Qiskit(qc, num_qubits)
    def swap_registers_Qiskit(qc: qiskit.QuantumCircuit, num_qubits):
        for j in range(num_qubits//2):
            qc.cx(j, num_qubits-j-1)
            qc.cx(num_qubits-j-1, j)
            qc.cx(j, num_qubits-j-1)
            # qc.barrier()
        return qc
    qc = qiskit.QuantumCircuit(num_qubits)
    qft_rotations_Qiskit(qc, num_qubits)
    swap_registers_Qiskit(qc, num_qubits)
    prob = qiskit.quantum_info.Statevector.from_instruction(qc).probabilities()
    return qc

import time
import numpy as np
times_qiskitss = []

qft_Qiskit(10) # Warm up the CPU
for qubit in range(3, 31):

    print("Qubit: ", qubit)
    time_qiskits = []
    for k in range(0, 5):
        
        start = time.time()
        qft_Qiskit(qubit)
        end = time.time()
        time_qiskits.append(end-start)
    times_qiskitss.append(np.mean(time_qiskits))
    np.savetxt("./result/qft_qiskit3_30.txt", times_qiskitss)

