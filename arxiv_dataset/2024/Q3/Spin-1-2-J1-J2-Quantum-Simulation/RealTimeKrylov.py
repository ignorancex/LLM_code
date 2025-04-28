
import numpy as np
import scipy
from qiskit.quantum_info import SparsePauliOp

def neel_order(dim):
    L = dim[0] * dim[1]
    neel_op = SparsePauliOp(('I' * L), coeffs=[0])

    for i in range(L):
        x, y = i % dim[0], i // dim[0]
        sign = (-1) ** (x + y)
        label = ['I'] * L
        label[i] = 'Z'
        neel_op += SparsePauliOp(''.join(label), coeffs=[sign])

    neel_op = neel_op.simplify()
    neel_op /= L  # Normalize the Néel operator

    return (neel_op @ neel_op).simplify()

def dimer_order(dim):
    Lx, Ly = dim
    num_spins = Lx*Ly
    dimer_op = SparsePauliOp(('I' * num_spins), coeffs=[0])
    normalization = 0

    for x in range(0, Lx//2, 2):
        for y in range(Ly):
            i = y * Ly + x
            j = ((x + 1) % Lx) + y * Ly
            sign = (-1)**(x)
            label = ['I'] * num_spins
            label[i] = label[j] = 'X'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Y'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Z'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            normalization += 1

    dimer_op.simplify()

    for x in range(0, Lx//2, 3):
        for y in range(Ly):
            i = y * Ly + x
            j = ((x + 2) % Lx) + y * Ly
            sign = (-1)**(x)
            label = ['I'] * num_spins
            label[i] = label[j] = 'X'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Y'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Z'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            normalization += 1

    dimer_op.simplify()

    for y in range(0, Ly//2, 2):
        for x in range(Lx):
            i = y * Ly + x
            j = ((y + 1) % Ly)*Ly + x
            sign = (-1)**(y)
            label = ['I'] * num_spins
            label[i] = label[j] = 'X'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Y'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Z'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            normalization += 1

    dimer_op.simplify()

    for y in range(0, Ly//2, 3):
        for x in range(Lx):
            i = y * Ly + x
            j = ((y + 2) % Ly)*Ly + x
            sign = (-1)**(y)
            label = ['I'] * num_spins
            label[i] = label[j] = 'X'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Y'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            label[i] = label[j] = 'Z'
            dimer_op += sign * SparsePauliOp(''.join(label), coeffs=[sign])
            normalization += 1

    dimer_op.simplify()
    dimer_op /= normalization
    return (dimer_op @ dimer_op).simplify()

def spin_order(dim):
    L = dim[0] * dim[1]
    spin_corr = SparsePauliOp(('I' * L), coeffs=[0])

    for i in range(L):
        for j in range(i+1, L):
            x1, y1 = i % dim[0], i // dim[0]
            x2, y2 = j % dim[0], j // dim[0]

            if abs(x1 - x2) + abs(y1 - y2) == 1:  # Nearest neighbors
                label = ['I'] * L
                label[i] = 'Z'
                label[j] = 'Z'
                spin_corr += SparsePauliOp(''.join(label), coeffs=[1])

    spin_corr = spin_corr.simplify()
    spin_corr /= L  # Normalize the spin correlation operator

    return spin_corr

def spin_correlation(dim, max_distance=None):
    L = dim[0] * dim[1]
    spin_corr = SparsePauliOp(('I' * L), coeffs=[0])

    for i in range(L):
        for j in range(i+1, L):
            x1, y1 = i % dim[0], i // dim[0]
            x2, y2 = j % dim[0], j // dim[0]

            distance = abs(x1 - x2) + abs(y1 - y2)
            if max_distance is None or distance <= max_distance:
                label = ['I'] * L
                label[i] = 'Z'
                label[j] = 'Z'
                coeff = 1 / (distance ** 2)  # Weight coefficient by inverse square of distance
                spin_corr += SparsePauliOp(''.join(label), coeffs=[coeff])

    spin_corr = spin_corr.simplify()
    spin_corr /= L  # Normalize the spin correlation operator

    return spin_corr

dim = [4,4]
neel_ham = neel_order(dim).to_matrix(sparse=True)
dimer_ham = dimer_order(dim).to_matrix(sparse=True)
spin_ham = spin_order(dim).to_matrix(sparse=True)
spin_corr_ham = spin_correlation(dim).to_matrix(sparse=True)

#local, global, neel, dimer
expectations = [spin_ham, spin_corr_ham, neel_ham, dimer_ham]

def construct_hamiltonian(j1, j2, grid):
  def nearest_neighbor(grid, i, j):
    i, j = i % len(grid[0]), j % len(grid)
    look_at = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    result = []
    for element in look_at:
      dx, dy = element
      result.append([(i + dx) % len(grid[0]), (j + dy) % len(grid)])
    return result

  def next_nearest_neighbor(grid, i, j):
    look_at = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    result = []
    for element in look_at:
      dx, dy = element
      result.append([(i + dx) % len(grid[0]), (j + dy) % len(grid)])
    return result

  def generate_dot_product(grid, term, idxA, idxB):
    operation_template = ['I' for element in range(len(grid[0]) * len(grid))]
    dot_product = SparsePauliOp(('I' * len(grid[0]) * len(grid)), coeffs=[0])
    for direction in ['X', 'Y', 'Z']:
      operation = operation_template
      operation[idxA], operation[idxB] = direction, direction
      dot_product += SparsePauliOp("".join(operation), coeffs=[term])
    return dot_product


  hamilonian = SparsePauliOp(('I' * len(grid[0]) * len(grid)), coeffs=[0])
  for i in range(len(grid[0])):
    for j in range(len(grid)):
      n_neighbors = nearest_neighbor(grid, i, j)
      nn_neighbors = next_nearest_neighbor(grid, i, j)

      for neighbor in n_neighbors:
        idxA = (j * len(grid)) + i
        idxB = (neighbor[1] * len(grid)) + neighbor[0]
        hamilonian += generate_dot_product(grid, j1, idxA, idxB)
      for neighbor in nn_neighbors:
        idxA = (j * len(grid)) + i
        idxB = (neighbor[1] * len(grid)) + neighbor[0]
        hamilonian += generate_dot_product(grid, j2, idxA, idxB)

  return hamilonian.simplify()

def modified_gram_schmidt(V):
    n, k = V.shape
    Q = np.zeros((n, k), dtype=complex)  # Orthogonalized vectors
    for i in range(k):
        q = V[:, i]
        for j in range(i):
            q -= np.vdot(Q[:, j], V[:, i]) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)
    return Q


for k in [0, 0.2, 0.4, 0.5, 0.56, 0.8, 1.0]:
    for l in range(2):
        random_vec = l >= 1
        print("Testing k: ", k, " Iteration: ", l, " Random Vec: ", random_vec)
        H_op = construct_hamiltonian(1.0, k, [[1/2 for i in range(4)] for j in range(4)])
        H_op = H_op.simplify()

        sparse_H = H_op.to_matrix(sparse=True)

        if(random_vec):
            v = np.random.rand(2**16)
            v /= np.linalg.norm(v)
        else:
            v = np.array([1 if bin(i)[2:] == "1010101010101010" else 0 for i in range(2**16)])
        v_csc = scipy.sparse.csc_matrix(v).T

        krylov_dim = 300
        dt = 1

        krylov_vectors = []
        import random
        for multiplier in range(krylov_dim):
            scaled_matrix = -1.0j * multiplier * 0.05 * sparse_H
            # Directly compute the product of expm and vector without forming expm
            result_vector = scipy.sparse.linalg.expm_multiply(scaled_matrix, v_csc)
            krylov_vectors.append(result_vector)  # result_vector is already in dense format by default

        import scipy.sparse as sp

        import numpy as np
        import scipy.sparse as sp

        # Assuming krylov_vectors is a list of sparse CSC matrices (each column vector of K_matrix)
        # Convert the list of vectors into a dense matrix for the orthonormalization process
        dense_K_matrix = np.hstack([vector.toarray() for vector in krylov_vectors])
        orthonormal_krylov_vectors = modified_gram_schmidt(dense_K_matrix)

        # Convert each orthonormal vector back to CSC format and stack them horizontally
        K_matrix = sp.hstack([sp.csc_matrix(orthonormal_krylov_vectors[:, i].reshape(-1, 1)) for i in range(krylov_dim)])

        # Multiply all vectors by sparse_H at once
        HK_matrix = sparse_H.dot(K_matrix)

        # Initialize the Hermitian matrix H_matrix_krylov of dimension krylov_dim x krylov_dim
        H_matrix_krylov = np.zeros((krylov_dim, krylov_dim), dtype=complex)

        # Compute the upper triangular part of the matrix
        for i in range(krylov_dim):
            for j in range(krylov_dim):
                element = K_matrix[:, i].conj().T.dot(HK_matrix[:, j])
                H_matrix_krylov[i, j] = element[0, 0] if element.size == 1 else element.toarray()[0, 0]

        import scipy.sparse as sp

        # Assuming 'K_matrix' has already been formed and is a sparse matrix where each column is a Krylov vector
        # Compute the similarity matrix by multiplying the transpose of K_matrix with K_matrix
        S_matrix_krylov = (K_matrix.conj().T).dot(K_matrix).toarray()

        import math
        import numpy as np
        import scipy as sp

        def solve_regularized_gen_eig(h: np.ndarray, s:np.ndarray, threshold: float, k: int =1, return_dimn: bool = False):
            s_vals, s_vecs = sp.linalg.eigh(s)
            s_vecs = s_vecs.T
            good_vecs = np.array([vec for val, vec in zip(s_vals, s_vecs) if val > threshold])
            h_reg = good_vecs.conj() @ h @ good_vecs.T
            s_reg = good_vecs.conj() @ s @ good_vecs.T
            if k==1:
                if return_dimn:
                    return sp.linalg.eigh(h_reg, s_reg)[0][0], sp.linalg.eigh(h_reg, s_reg), len(good_vecs)
                else:
                    return sp.linalg.eigh(h_reg, s_reg)[0][0], sp.linalg.eigh(h_reg, s_reg)
            else:
                if return_dimn:
                    return sp.linalg.eigh(h_reg, s_reg)[0][:k], sp.linalg.eigh(h_reg, s_reg), len(good_vecs)
                else:
                    return sp.linalg.eigh(h_reg, s_reg)[0][:k], sp.linalg.eigh(h_reg, s_reg)

        gnd_en_circ_est_list = []
        last_one = None
        for d in range(krylov_dim, krylov_dim+1):
            # Solve generalized eigenvalue problem
            gnd_en_circ_est, vectors = solve_regularized_gen_eig(H_matrix_krylov[:d, :d], S_matrix_krylov[:d, :d], threshold=1e-8)
            gnd_en_circ_est_list.append(gnd_en_circ_est)
            last_one = vectors

        import scipy.sparse as sp
        import numpy as np

        print("Energy: ", last_one[0][0])

        # Assuming 'krylov_vectors', 'expectations', 'S_matrix_krylov', and 'last_one' are defined elsewhere

        for expectation in expectations:

            # Multiply all vectors by sparse expectation at once
            HK_matrix = expectation.dot(K_matrix)

            # Initialize the Hermitian matrix expectation_matrix_krylov of dimension krylov_dim x krylov_dim
            expectation_matrix_krylov = np.zeros((krylov_dim, krylov_dim), dtype=complex)

            # Compute the upper triangular part of the matrix
            for i in range(krylov_dim):
                for j in range(krylov_dim):
                    element = K_matrix[:, i].conj().T.dot(HK_matrix[:, j])
                    expectation_matrix_krylov[i, j] = element[0, 0] if element.size == 1 else element.toarray()[0, 0]

            corrected_vector = (np.array(last_one[1][0]))
            corrected_expectation_value = corrected_vector.conj().T.dot(expectation_matrix_krylov).dot(corrected_vector)
            print("Expectation value:", corrected_expectation_value)