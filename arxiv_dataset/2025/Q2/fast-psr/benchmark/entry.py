import qimax.utilities
import qimax.tensor
import qimax.gate
import qimax.converter
import numpy as np, qiskit, re
import time
import qimax.gradient
from qoop.core.random_circuit import generate_with_pool


start = time.time()
num_qubits = list(range(2, 4))
depths = list(range(2, 10))
time_psr = np.zeros((len(num_qubits), len(depths)))
time_proposed_psr = np.zeros((len(num_qubits), len(depths)))
time_qiskit = np.zeros((len(num_qubits), len(depths)))
for i, num_qubit in enumerate(num_qubits):
    for j, depth in enumerate(depths):
            #print(num_qubit, depth)
            time_psrs, time_proposed_psrs, time_qiskits = [], [], []
            for k in range(5):
                qc = generate_with_pool(num_qubit, depth)
                start = time.time()
                result1 = qimax.gradient.psr(qc)
                end = time.time()
                time_psrs.append(end - start)
                start = time.time()
                result1 = qimax.gradient.proposed_psr(qc)
                end = time.time()
                time_proposed_psrs.append(end - start)
                start = time.time()
                result1 = qimax.gradient.qiskit(qc)
                end = time.time()
                time_qiskits.append(end - start)
            time_psr[i, j] = np.mean(time_psrs)
            time_proposed_psr[i, j] = np.mean(time_proposed_psrs)
            time_qiskit[i, j] = np.mean(time_qiskits)
# np.savetxt(f"result/time_psr.csv", time_psr, delimiter=",")
# np.savetxt(f"result/time_proposed_psr.csv", time_proposed_psr, delimiter=",")
# np.savetxt(f"result/time_qiskit.csv", time_qiskit, delimiter=",")
print(time.time() - start)