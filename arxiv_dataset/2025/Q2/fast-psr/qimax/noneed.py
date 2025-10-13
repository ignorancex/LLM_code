

def circuit_to_gates(qc: qiskit.QuantumCircuit):
    gates = []
    for instr, qargs, _ in qc.data:
        qubit_indices = [q.index for q in qargs]
        name = instr.name.upper()
        gates.append(gate.Gate(name[1:] if name[0] == 'C' else name, indices = qubit_indices, param = np.squeeze(instr.params)))
    return gates