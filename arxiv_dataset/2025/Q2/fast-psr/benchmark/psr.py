
def psr_qiskit(num_qubits: int):
    import qiskit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
    import numpy as np
    thetas = qiskit.circuit.ParameterVector('theta', num_qubits * 3)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for j in range(num_qubits):
        qc.h(j)
    for j in range(num_qubits):
        qc.cx(j, (j + 1) % num_qubits)
    k = 0
    for j in range(num_qubits):
        qc.rx(thetas[k], j)
        qc.ry(thetas[k + 1], j)
        qc.rz(thetas[k + 2], j)
        k += 3
    H = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
    params = [np.random.uniform(0, 2*np.pi, 3*num_qubits)]
    estimator = Estimator()
    grad = ParamShiftEstimatorGradient(estimator).run(qc, H, params).result().gradients
    return grad



def psr_pennylane(num_qubits: int):
    import pennylane as qml
    from pennylane import numpy as pnp
    import numpy as np
    dev = qml.device("default.qubit")
    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def circuit(params, num_qubits):
        for j in range(num_qubits):
            qml.Hadamard(wires=j)
        for j in range(num_qubits):
            qml.CNOT(wires=[j, (j + 1) % num_qubits])
        k = 0
        for j in range(num_qubits):
            qml.RX(params[k], wires=j)
            qml.RY(params[k+1], wires=j)
            qml.RZ(params[k+2], wires=j)
            k += 3
        H = qml.PauliZ(0)
        for j in range(1, num_qubits):
            H = H @ qml.PauliZ(j)
        return qml.expval(H)
    params = pnp.array(np.random.uniform(0, 2*np.pi, 3*num_qubits), requires_grad=True)
    grad = qml.jacobian(circuit)(params, num_qubits)
    return grad
# psr_pennylane(num_qubits = 15)

def psr_qulacs(num_qubits: int):    
    from qulacs import ParametricQuantumCircuit, GradCalculator, Observable
    import numpy as np
    observable = Observable(num_qubits)
    observable_str = " ".join([f"Z {i}" for i in range(num_qubits)])
    observable.add_operator(1.0, observable_str)
    circuit = ParametricQuantumCircuit(num_qubits)

    theta = np.random.uniform(0, 2*np.pi, 3*num_qubits)
    for j in range(num_qubits):
        circuit.add_H_gate(j)
    for j in range(num_qubits):
        circuit.add_CNOT_gate(j, (j + 1) % num_qubits)
    k = 0
    for j in range(num_qubits):
        circuit.add_parametric_RX_gate(j, theta[k])
        circuit.add_parametric_RY_gate(j, theta[k + 1])
        circuit.add_parametric_RZ_gate(j, theta[k + 2])
        k += 3
    gcalc = GradCalculator()
    return (np.real(gcalc.calculate_grad(circuit, observable, theta)))
# psr_qulacs(15)