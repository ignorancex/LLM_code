import numpy as np
gate = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "RX": lambda theta: np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    ),
    "RY": lambda theta: np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    ),
    "RZ": lambda theta: np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128
    ),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
}

import qiskit

num_qubits = 3
qc = qiskit.QuantumCircuit(num_qubits)
qc.rz(0.3, 0)
qc.rz(0.4, 1)
qc.cx(0, 1)
qc.h(2)
qc.h(0)
qc.rx(0.5, 1)
qc.h(2)
qc.cx(0, 1)
qc.rz(0.5, 0)
qc.h(2)
from qimax import converter, circuit, splitter


qcs = splitter.qc_to_qcs(qc)
gatess1 = converter.qcs_to_gatess(qcs)


gatess = [[0 for _ in range(num_qubits)] for _ in range(len(gatess1))]
for index, gates in enumerate(gatess1):
    slots = np.ones(num_qubits)
    print('---')
    print(gates)
    for gate in gates:
        if gate[0] == 'CX':
            gatess[index][0] = ('CX', gate[1], gate[2])
        else:
            slots[gate[2][0]] = 0
            gatess[index][gate[2][0]] = (gate[0], gate[1], gate[2])
    if gate == 'CX':
        continue
    for j, is_active in enumerate(slots):
        if is_active:
            gatess[index][j] = (['I', -999, [j]])

print('xxxx')  
for u in gatess:
    print(u)