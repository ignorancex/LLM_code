import qutip as qt
from qutip.core.gates import rx, ry, hadamard_transform
import numpy as np
import jax.numpy as jnp
import jax

from itertools import chain 
from jax import random
import optax
from tqdm.auto import tqdm
import time
from numpy.random import default_rng
from itertools import product, chain


def cnot_gate(n, control, target):

    """
    Create a CNOT gate for an n-qubit system with the specified control and target qubits.

    Parameters:
        n (int): Number of qubits in the system.
        control (int): Index of the control qubit (0-based).
        target (int): Index of the target qubit (0-based).

    Returns:
        Qobj: The CNOT gate for the n-qubit system.
    """
    # Identity operator for a single qubit
    I = qt.qeye(2)
    # Pauli-X gate for a single qubit
    X = qt.sigmax()

    # Create the projection operators for the control qubit
    P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()  # |0><0|
    P1 = qt.basis(2, 1) * qt.basis(2, 1).dag()  # |1><1|

    # Start building the full CNOT gate
    # Part 1: When the control qubit is |0>, no change
    CNOT = qt.tensor(
        *[
            I if i != control else P0
            for i in range(n)
        ]
    ) 
    # Part 2: When the control qubit is |1>, apply X to the target qubit 
    CNOT += qt.tensor(
        *[
            I if i != control and i != target
            else (P1 if i == control else X)
            for i in range(n)
        ]
    )

    return CNOT

def parse_circuit_to_qobj(text_circuits, n):
    """
    Converts a list of text-based quantum circuits into Qutip unitary operators.

    Args:
        text_circuits (list of str): List of text circuit descriptions.
        n (int): Number of qubits.

    Returns:
        list of Qobj: List of Qutip unitary operators of dimension [[2]^n, [2]^n].
    """
    unitary_list = []

    for circuit in text_circuits:
        U = qt.qeye([2] * n)  # Start with identity matrix
        
        # Extract operations from the text
        operations = circuit.split(")(")  
        operations = [op.strip("()") for op in operations if op.strip("()")]  # Remove empty entries

        for op in operations:
            gate_info = op.split(":")
            if len(gate_info) < 2:
                continue  # Skip invalid formats

            gate_name = gate_info[0]
            qubit_indices = list(map(int, gate_info[1].split(",")))

            # Apply the correct gate
            if gate_name == "RX90":
                U = qt.tensor([rx(np.pi/2) if i == qubit_indices[0] else qt.qeye(2) for i in range(n)]) * U
            elif gate_name == "RY90":
                U = qt.tensor([ry(np.pi/2) if i == qubit_indices[0] else qt.qeye(2) for i in range(n)]) * U
            elif gate_name == "H":
                U = qt.tensor([hadamard_transform(N=1) if i == qubit_indices[0] else qt.qeye(2) for i in range(n)]) * U
            elif gate_name == "CNOT":
                U = cnot_gate(n, qubit_indices[0], qubit_indices[1]) * U
            else:
                raise ValueError(f"Unsupported gate: {gate_name}")

        unitary_list.append(U)

    return unitary_list


# Prepare State
def prepare_state(initial_text,N):
    U_state_prep=parse_circuit_to_qobj([initial_text], N)
    initial_state= qt.tensor([qt.basis(2,0)]*N)
    rho_ideal=U_state_prep[0]*initial_state
    rho_ideal=rho_ideal*rho_ideal.dag()
    return rho_ideal

def flatten_list(nested_list):
    """Flattens a list of lists into a single list using itertools.chain."""
    return list(chain(*nested_list))


def block_seperator(rho , selective_blocks, N):
    """
    Extracts and sums specific block structures from a given density matrix based on selective binary representations.
    
    Args:
        rho (numpy.ndarray): Input density matrix of shape (2^N, 2^N).
        selective_blocks (list of int): List of integers representing selective blocks to extract (must be < 2^N).
        N (int): Number of qubits in the system.

    Returns:
        qutip.Qobj: Resulting matrix as a Qobj with dimension (2^N, 2^N), representing the sum of selected blocks.
    """
    
    assert all(0 <= block < 2**N for block in selective_blocks), "All entries in selective_blocks must be less than 2**N"
    
    binary_reps = [format(block, f'0{N}b') for block in selective_blocks]
    
    pauli_observables = []
    for binary in binary_reps:
        operators = [qt.qeye(2) if bit == '0' else qt.sigmax() for bit in binary]
        pauli_observables.append(qt.tensor(*operators))
    
    total_observable = sum(obs.full() * rho for obs in pauli_observables)
    
    return qt.Qobj(total_observable, dims=[[2]*N, [2]*N])




@jax.jit
def data_predict_from_rho(rho, circuits, shots):
    """
    Predict measurement outcomes from a given quantum state (rho) and a set of quantum circuits.
    
    Args:
        rho (jnp.array): Density matrix representing the quantum state.
        circuits (jnp.array): Array of quantum circuit matrices applied to the state.
        shots (int ): Number of measurement shots to scale probabilities.

    Returns:
        jnp.array: Predicted measurement probabilities scaled by shots for each circuit shape(circuit,comp basis).
    """
    # Compute predicted measurement probabilities for each circuit
    def compute_for_circuit(circuit):
        return jnp.diag(jnp.dot(jnp.dot(circuit, rho), circuit.T.conj())) * shots
    
    # Use vmap to apply computation across all circuits efficiently
    data_predict = jax.vmap(compute_for_circuit)(circuits)
    
    return jnp.real(data_predict)

def data_predict_from_rho_sampled(rho, circuits, shots):
    """
    Predict measurement outcomes from a quantum state (rho) and circuits,
    sampling finite-shot counts from the probability distribution.

    Args:
        rho (jnp.array): Density matrix.
        circuits (jnp.array): Array of circuit unitaries.
        shots (int): Number of measurement shots.

    Returns:
        jnp.array: Sampled counts (integer array) for each circuit.
    """

    def compute_probs(circuit):
        probs = jnp.real(jnp.diag(circuit @ rho @ circuit.T.conj()))
        probs = probs / jnp.sum(probs)  # normalize to sum to 1
        return probs

    # Step 1: compute all probabilities with vmap
    probs_all = jax.vmap(compute_probs)(circuits)

    # Step 2: sample counts with NumPy loop
    counts_list = []
    for probs in np.array(probs_all):  # convert to numpy array
        counts = np.random.multinomial(shots, probs)
        counts_list.append(counts)

    return jnp.array(counts_list)


def random_matrix_jax(N, key=random.PRNGKey(0)):
    """
    Generate a random (non-Hermitian) matrix using JAX for an N-qubit system.
    
    Parameters:
        N (int): Number of qubits.
        key: JAX PRNG key.
        
    Returns:
        jnp.array: random complex matrix
    """
    # Total Hilbert space dimension for N qubits
    dim = 2**N
    
    # Generate random real and imaginary parts using JAX
    key_real, key_imag = random.split(key)
    real_part = random.uniform(key_real, (dim, dim))
    imag_part = random.uniform(key_imag, (dim, dim)) * 1j
    
    # Combine real and imaginary parts
    random_matrix = real_part + imag_part
    
    
    return random_matrix


# Compute the density matrix rho = A^T A / tr(A^T A)
@jax.jit
def density_matrix(A):
    AtA = jnp.dot(A.T.conj(), A)
    rho = AtA / jnp.trace(AtA)
    return rho 

# Define the log loss function (user-defined)
@jax.jit
def log_loss_function(A,data,circuits,shots):
    rho = density_matrix(A)
    predicted_data = data_predict_from_rho(rho,circuits,shots=1)
    return -jnp.sum(data * jnp.log(predicted_data+1e-8))/shots

def gd_chol_rank(data, rho_or, ops_jnp,shots, params: optax.Params, iterations: int,  batch_size: int,
            lr=2e-1, decay = 0.999, lamb:float =0.00001, batch=False, tqdm_off=False):
  """
  Function to do the GD-Chol.
  Return:
    params1: The reconstructed density matrix
    fidelities_GD: A list with the fidelities values per iteration
    timel_GD: A list with the value of the time per iteration
    loss1: A list with the value of the loss function per iteration

  Input:
    data: the expected value of the original density matrix
    rho_or: original density matrix, to calculate the fidelity
    ops_jnp: POVM in jnp array
    params: Ansatz, any complex matrix T (not necessary the lower triangular)
    iterations: number of iterations for the method
    batch_size: batch size
    lr: learning rate
    decay: value of the decay of the lr (defined in given optimizer)
    lamb: hyperparameter l1 regularization
    batch: True to have mini batches, False to take all the data
    tqdm_off: To show the iteration bar. True is to desactivate (for the cluster)
    
  """
  convergence_tol = 1e-10  # minimal change considered as progress
  patience = 50           # number of allowed stagnant steps before stopping
  no_improve_counter = 0
  
  start_learning_rate = lr
  # Exponential decay of the learning rate.
  scheduler = optax.exponential_decay(
      init_value=start_learning_rate, 
      transition_steps=iterations,
      decay_rate=decay)
  # Combining gradient transforms using `optax.chain`.
  gradient_transform = optax.chain(
      optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
      optax.scale_by_adam(),  # Use the updates from adam.
      optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
      # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
      optax.scale(-1.0)
  )
  

  loss1 = []
  fidelities_GD = []
  timel_GD = []
  #par_o = jnp.matmul(jnp.conj(params.T),params)/jnp.trace(jnp.matmul(jnp.conj(params.T),params))
  #fidelities_GD.append(qtp.fidelity(rho_or, qtp.Qobj(par_o)))
  #loss1.append(float(cost(params, jnp.asarray(data), ops_jnp, lamb)))
  opt_state = gradient_transform.init(params)
  num_me = len(data)
  # opt_state = optimizer.init(params)
  if not tqdm_off:
    pbar_GD = tqdm(range(iterations)) 
  
  @jax.jit
  def step(params, opt_state, data, ops_jnp):
    grad_f = jax.grad(log_loss_function, argnums=0)(params,data,ops_jnp,shots)
    grads = jnp.conj(grad_f)           # do a conjugate, if not can create some problems
    # updates, opt_state = optimizer.update(grads, opt_state, params)
    updates, opt_state = gradient_transform.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state
  

  tot_time = 0
  for i in tqdm(range(iterations), disable=tqdm_off):
    start = time.time()
    if batch:
        rng = default_rng()
        indix = rng.choice(num_me, size=batch_size, replace=False)
        # indix = np.random.randint(0, num_me, size=[batch_size])
        data_b = jnp.asarray(data[[indix]].flatten())
        ops2 = ops_jnp[indix]
    else: 
        ops2 = ops_jnp
        data_b = data
    params, opt_state = step(params, opt_state,data_b, ops2)
    #params = rho_cons(params)
    end = time.time()
    timestep = end - start
    tot_time += timestep
    timel_GD.append(tot_time)    
    
    par1 = jnp.matmul(jnp.conj(params.T),params)/jnp.trace(jnp.matmul(jnp.conj(params.T),params))
    loss1.append(float(log_loss_function(params,data,ops_jnp,shots)))
    f = qt.fidelity(rho_or, qt.Qobj(np.array(par1)))
    fidelities_GD.append(f)
    

    #timel_GD.append(end - start)  
    if not tqdm_off:
        pbar_GD.set_description("Fidelity GD-chol-rank {:.4f}".format(f))
        pbar_GD.update() 
        
    if len(loss1) > 10:
        if abs(loss1[-1] - loss1[-2]) < convergence_tol:
            no_improve_counter += 1
        else:
            no_improve_counter = 0
    
    if no_improve_counter >= patience:
        #print(f"Early stopping at iteration {i} due to convergence.")
        break

  params1 = jnp.matmul(jnp.conj(params.T),params)/jnp.trace(jnp.matmul(jnp.conj(params.T),params))
  return params1, fidelities_GD, timel_GD, loss1



def process_data(data,unitaries_jnp,selective_blocks,shots,N,rho_ideal=None):
    A= random_matrix_jax(N)
    iterations = 100*N*3
    if rho_ideal!=None:
        rho_ideal = qt.Qobj(np.array(rho_ideal.full())) #Necessary conversion
    else: 
        rho_ideal=qt.rand_dm(2**N)
    rho, fidelities, time_GD, loss= gd_chol_rank(data,rho_ideal,unitaries_jnp,shots,A, iterations, 100, lr=0.1, decay = 0.999, lamb=0.00001, batch=False, tqdm_off=True)

    return block_seperator(np.array(rho),selective_blocks,N) 
