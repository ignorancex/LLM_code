import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def create_block_encoding(a):
    M = np.array([[a, -1], [-1, a]])
    kappa = a + 1
    C = M / kappa

    # Eigen decomposition for matrix square root
    eigvals, eigvecs = np.linalg.eigh(C)
    S_eigvals = np.sqrt(np.maximum(1 - eigvals**2, 0))  # Ensure real values
    S = eigvecs @ np.diag(S_eigvals) @ eigvecs.T

    # Construct unitary block encoding
    top = np.hstack((C, -S))
    bottom = np.hstack((S, C))
    return np.vstack((top, bottom))

def circuit_before_aa(n, a, marked_state, wires):
    """Circuit part before amplitude amplification (U_C)"""
    # Prepare initial states
    qml.Hadamard(wires=wires[1])
    for i in wires[2:]:
        qml.Hadamard(wires=i)

    # Apply controlled oracle
    diag = np.ones(2**(n+2))
    index0 = (0 << (n+1)) | (1 << n) | marked_state
    index1 = (1 << (n+1)) | (1 << n) | marked_state
    diag[index0] = -1
    diag[index1] = -1
    qml.DiagonalQubitUnitary(diag, wires=wires)

    # Apply block encoding
    Uc = create_block_encoding(a)
    qml.QubitUnitary(Uc, wires=wires[:2])

def circuit(n, a, marked_state, wires, apply_aa=False, aa_steps=1):
    # First apply the base circuit (U_C)
    circuit_before_aa(n, a, marked_state, wires)
    
    if apply_aa:
        # Define the good state: ancilla=0 (wire0)
        def oracle():
            # Mark states where ancilla is |0> (flip phase)
            qml.PauliX(wires=wires[0])
            qml.PauliZ(wires=wires[0])
            qml.PauliX(wires=wires[0])
        
        # Diffusion operator: reflection around initial state
        def diffusion():
            # Step 1: Apply U_C^\dagger
            qml.adjoint(circuit_before_aa)(n, a, marked_state, wires)
            
            # Step 2: Reflect around |0> (U_0 = I - 2|0><0|)
            diag = np.ones(2**len(wires))
            diag[0] = -1  # Flip phase of |0> state
            qml.DiagonalQubitUnitary(diag, wires=wires)
            
            # Step 3: Apply U_C again
            circuit_before_aa(n, a, marked_state, wires)
        
        # Apply amplitude amplification steps
        for _ in range(aa_steps):
            oracle()
            diffusion()

def simulate(n, a, marked_state, apply_aa=False, aa_steps=1):
    num_wires = n + 2
    wires = list(range(num_wires))
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def qnode():
        circuit(n, a, marked_state, wires, apply_aa, aa_steps)
        return qml.state()

    state = qnode()
    probs = np.abs(state)**2

    # Calculate probability of ancilla=0
    p0 = np.sum(probs[:2**(num_wires-1)])

    # Calculate joint probability (ancilla=0 and data=marked_state)
    p_joint = 0.0
    for c in [0, 1]:
        idx = (0 << (n+1)) | (c << n) | marked_state
        p_joint += probs[idx]

    p_marked_given_0 = p_joint / p0 if p0 > 0 else 0
    return p0, p_marked_given_0

# Parameters
n_values = list(range(2, 10))
a_values = [(n**2-1)/n**2 for n in n_values]
results_no_aa = []
results_aa = []

# Run simulations
for i, n in enumerate(n_values):
    marked_state = np.random.randint(0, 2**n)
    a = a_values[i]
    
    # Without amplitude amplification
    p0, p_marked = simulate(n, a, marked_state, apply_aa=False)
    results_no_aa.append((n, p0, p_marked))
    
    # With amplitude amplification (using n steps as in paper)
    p0_aa, p_marked_aa = simulate(n, a, marked_state, apply_aa=True, aa_steps=n)
    results_aa.append((n, p0_aa, p_marked_aa))

# Plotting function
def plot_results(results, title_suffix=""):
    n_vals = [r[0] for r in results]
    p0_vals = [r[1] for r in results]
    p_marked_vals = [r[2] for r in results]


    plt.plot(n_vals, p0_vals, 'o-', label='$P_0$ (ancilla=0)')
    plt.plot(n_vals, p_marked_vals, 's-', label='$P$(marked|ancilla=0)')
    plt.plot(n_vals, [1/n for n in n_vals], '^--', label='$1/n$')
    plt.plot(n_vals, [1/np.sqrt(2**n) for n in n_vals], 'x--', label='$1/\sqrt{N}$')
    plt.plot(n_vals, a_values, 'd--', label='$a$')
    plt.xlabel('Number of Data Qubits (n)')
    plt.ylabel('Probability')
    plt.legend()
    plt.title(f'Unitary Search with Ancilla {title_suffix}')
    plt.xticks(n_vals)
    plt.grid(True)
  

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_results(results_no_aa, "(without AA)")
plt.subplot(1, 2, 2)
plot_results(results_aa, "(with AA for ancilla=0)")
plt.show()