import math

def run(qc):
    mch(qc, [0, 4], 7)  # M1

    qc.barrier()

    swap_registers(qc,10)
    #qft_rotations(qc,10)
    for qubit in range(10):
        qc.h(qubit)
        p = 1
        qft_rotations(qc,qubit,p)

    qc.barrier()

    qc.measure([0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9])

def mch(qc, c_q, t_q):
    qc.ry(math.pi/4,t_q)
    qc.mct(c_q,t_q)
    qc.ry(-math.pi/4,t_q)
    return qc

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft_rotations(circuit, qubit, p):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    for j in range(qubit + 1, 10):
        circuit.cu1(math.pi/2**(p), qubit, j)
        p += 1
    circuit.barrier()
    return circuit





