import qiskit
import time
import numpy as np
from qiskit_aer import Aer
import ansatz
import pandas as pd

def quanv_cpu(num_qubits, function_name):
    qc = qiskit.QuantumCircuit(num_qubits) 
    qc = getattr(ansatz, function_name)(qc)
    qc = qiskit.transpile(qc, basis_gates=['h', 's', 'cx', 'rx', 'ry', 'rz'], optimization_level=3)
    state = qiskit.quantum_info.Statevector.from_instruction(qc)
    return

for i in range(1, 10):
    quanv_cpu(10, 'quanvolutional1')  # Warm up the GPU
for num_qubits in range(29, 31):
    for i in range(1, 20):
        
        function_name = f'quanvolutional{i}'
        times = []
        for t in range(0, 2):
            start = time.time()
            quanv_cpu(num_qubits, function_name)
            end = time.time()
            times.append(end-start)
        record = {'num_qubits': num_qubits, 'quanvolutional': i, 'time': np.mean(times)}
        if 'results_df' not in locals():
            results_df = pd.DataFrame(columns=record.keys())
        results_df = pd.concat([results_df, pd.DataFrame([record])], ignore_index=True)
        results_df.to_csv(f'result/quanv_cpu29_30.csv', index=False)
        print(f'num_qubits: {num_qubits}, quanvolutional{i}, time: ', np.mean(times))
