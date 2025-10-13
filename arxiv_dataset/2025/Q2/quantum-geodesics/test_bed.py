from quantum_geodesics.boilerplate import kron, generate_pauli_signatures, get_pauli_dict, basis_constructor
import numpy as np

def test_hypothesis(n_qubits:int, n_bases:int):

    qubits = [np.atleast_2d(qubit).T for qubit in np.exp(1j*np.random.uniform(0, 2*np.pi, size=(n_qubits,2)))]

    coeffs = np.random.uniform(1e-3, 1, size=n_bases)*np.exp(1j*np.random.uniform(0, 2*np.pi, size=n_bases))

    bases = [basis for _, basis in generate_pauli_signatures(n_qubits=n_qubits)]

    selected_bases = list(str(blah) for blah in np.random.choice(bases, size=n_bases, replace=False))

    pauli_dict = get_pauli_dict()
    pauli_components = list(list(pauli_dict[sign] for sign in signature) for signature in selected_bases)
    
    full_operator = np.sum(
        [coeff*basis_constructor(pauli_component) for coeff, pauli_component in zip(coeffs, pauli_components)],
        axis=0
    )

    full_qubits = basis_constructor(qubits)

    old_solution = full_operator@full_qubits

    new_setup = list(coeff*basis_constructor(list(item@qubit for item, qubit in zip(component, qubits))) for coeff, component in zip(coeffs, pauli_components))

    new_solution = np.sum(
        new_setup,
        axis=0
    )

    print(np.allclose(new_solution, old_solution))
    print(np.abs(new_solution - old_solution) < 1e-6)

    print(np.column_stack((old_solution, new_solution)))


test_hypothesis(12, 10)