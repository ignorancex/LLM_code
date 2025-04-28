# Author: Bill Pengyuan Zhai. Harvard University. Yelin Group. Oct 2022

from Utils_torch_version import Network, get_nn_pairs, binary_basis, unpacknbits
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy
import scipy.linalg
import qiskit
import time
import torch
import math

# try another way of evolving
# The sparse H_OP is a sparse.csr matrix. Useful for calculating via scipy diagonalization. Does not involve qiskit classes
def initialize_sparse(conn_i, conn_j, a, b, c, d): # 
    print('debug a, b, c, d)',(a, b, c, d))
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sy = sparse.csr_matrix(np.array([[0 , -1j], [1j , 0]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    # A list of single x, y, z operator applied on the site i_site
    sx_list = []
    sy_list = []
    sz_list = []
    for i_site in range(N):
        x_ops = [id] * N
        y_ops = [id] * N
        z_ops = [id] * N
        x_ops[i_site] = sx
        y_ops[i_site] = sy
        z_ops[i_site] = sz
        X = x_ops[0]
        Y = y_ops[0]
        Z = z_ops[0]
        for j in range(1, N):
            X = sparse.kron(X, x_ops[j], 'csr')
            Y = sparse.kron(Y, y_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sy_list.append(Y)
        sz_list.append(Z)
    # H_zz = sparse.csr_matrix((2**N, 2**N))
    # H_xx = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_yy = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_xy = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_yx = sparse.csr_matrix(np.zeros((2**N, 2**N)), dtype=np.complex128)
    # H_z = sparse.csr_matrix(np.zeros((2**N, 2**N)),dtype=np.complex128)
    #H_zzzi = sparse.csr_matrix((2**N, 2**N),dtype=np.complex128)

    # H_zz = H_zz + sz_list[i] * sz_list[j]
    H_xx = sx_list[conn_i] @ sx_list[conn_j]
    H_yy = sy_list[conn_i] @ sy_list[conn_j]
    H_xy = sx_list[conn_i] @ sy_list[conn_j]
    H_yx = sy_list[conn_i] @ sx_list[conn_j]

    H_zzz_id = sparse.csr_matrix(np.eye(2**N), dtype=np.complex128) # This is a diagonal of real numbers
    for i in range(conn_i, conn_j):
        H_zzz_id = sz_list[i] @  H_zzz_id
    
    H_z = - (a/2) * sz_list[conn_i] - (b/2) * sz_list[conn_j]

    H = H_z + (  (1.0j*c/2) * H_xy - (1.0j*c/2) * H_yx + (d*(1.0j)/2) * H_xx + (d*(1.0j)/2) * H_yy  ) @ H_zzz_id
    
    print('H_z', H_z)
    print('H_xy',H_xy)
    print('H_yx',H_yx)
    print('H_xx',H_xx)
    print('H_yy',H_yy)

    print('debug H', H) # Somehow this doesn't come out to be Hermitian lol
    print('debug iH', 1j* H)
    return H.todense()


Ns = [4, 6, 8, 12]
reps = 1
avg_time_at_N_fermion, avg_time_at_N_exact = [], []
q_fidelities_all_runs, TVs_all_runs = [], []
for N in Ns:
    print('start N: ', N)
    conn_list = [p for p in get_nn_pairs(geometry=(N,)) if p != (N-1, 0) and p != (0, N-1)] # zero indexed and should not be periodic (not a closed circle)
    #conn_list = [[i, N-1] for i in range(N-1)]*2


    times_fermion = []
    times_exact = []
    for rep in range(reps):
        L = len(conn_list) # Number of layers

        # initiliaze the circuit
        circuit = Network(conn_list, N)

        x = torch.tensor([[1,0]*int(N/2)]) # The 1010101... single basis state
        init_state_numpy = qiskit.quantum_info.Statevector.from_label('10'*int(N/2)).data

        # Fix this, the parameters are defined differently now for the pytorch implementation
        params_m = torch.tensor(math.pi) * torch.rand((L, 4))
        
        print('params_m',params_m)

        circuit.manual_set_params(params_m)


        basis_m_n = torch.tensor(binary_basis(geometry=(N,)))
    
        amps = torch.zeros(2**N, dtype=torch.cfloat)
        ts = time.time()

        sums = torch.sum(basis_m_n, axis=1)
        print('sums', sums)
        diff = sums-torch.sum(x) # a list of differences in the number of Fermions
        print('diff', diff)

        indices = (diff == 0).nonzero().flatten() # these are the indices where probability can be nonzero
        print('indices', indices)
        # Prepare the batches
        
        n_batches = len(indices)//10 if len(indices)%10 == 0 else len(indices)//10+1
        for i in range(n_batches):
            y_batch = basis_m_n[indices[10*i : 10*(i+1)]]
            x_batch = x.repeat(y_batch.shape[0], 1) # a batch of 10

            print('y_batch, x_batch', (y_batch, x_batch))

            amp_batch = circuit.forward(y_batch, x_batch)
            print('debug amp_batch', amp_batch)
            # Now put these amp_batch values into correct positions
            amps[indices[10*i : 10*(i+1)]] = amp_batch

        print('amps', amps)
        amps = amps.detach().numpy()
        tf = time.time()
        times_fermion.append(tf - ts)


        ts = time.time()
        exp_iH_exact = np.eye(2**N)
        for l in range(L):
            conn = conn_list[l]
            a, b, c, d = params_m.detach().numpy()[l]
            H_exact = initialize_sparse(conn[0], conn[1], a, b, c, d)
            exp_iH_exact = (scipy.linalg.expm(-1.0j*H_exact))@exp_iH_exact # 1.0j or -1.0j?
            
        state_exact = np.matmul(exp_iH_exact, init_state_numpy[:,None]) 
        tf = time.time()
        times_exact.append(tf - ts)

        probs_exact = (np.abs(state_exact)**2).squeeze()
        # print('exact evolved state', state_exact)
        # print('exact evolved prob', probs_exact)

        probs_fermion = np.abs(amps)**2
        # print('fermion state', amps)
        # print('Fermion probs', probs_fermion)
        # print('sum Fermion probs', sum(probs_fermion))
        
        plt.plot(probs_exact, '^-')
        plt.plot(probs_fermion, 'x')
        plt.legend(['exact probs', 'Fermion_probs'])
        plt.title('(N, rep)='+str((N, rep)))
        plt.savefig('img_torch/(N, rep)='+str((N, rep))+'.png')
        plt.close()

        # calculate the quantum fidelity
        q_fidelity1 = np.abs((np.conjugate(amps).dot(state_exact[:,0])))**2
        #fidelity2 = np.abs((np.conjugate(state_exact[:,0]).dot(amps)))**2
        print('q_fidelity', q_fidelity1)

        print('debug probs_fermion shape', probs_fermion.shape)
        print('debug probs_exact shape', probs_exact.shape)

        print('diff', probs_fermion-probs_exact)
        tv = np.sum(np.abs(probs_fermion-probs_exact))
        print('tv', tv)

        q_fidelities_all_runs.append(q_fidelity1)
        TVs_all_runs.append(tv)

    avg_time_fermion = sum(times_fermion)/reps
    std_time_fermion = np.std(times_fermion)

    avg_time_exact = sum(times_exact)/reps
    std_time_exact = np.std(times_exact)


    avg_time_at_N_fermion.append(avg_time_fermion)
    avg_time_at_N_exact.append(avg_time_exact)

print('avg_time_at_N_fermion', avg_time_at_N_fermion)
print('avg_time_at_N_exact', avg_time_at_N_exact)
plt.plot(Ns, avg_time_at_N_fermion)
plt.plot(Ns, avg_time_at_N_exact)

plt.plot(Ns, avg_time_at_N_fermion+std_time_fermion, '+')
plt.plot(Ns, avg_time_at_N_fermion-std_time_fermion, '-')


plt.plot(Ns, avg_time_at_N_exact+std_time_exact, '^')
plt.plot(Ns, avg_time_at_N_exact-std_time_exact, 'v')
plt.legend(['avg_time_at_N_fermion', 'avg_time_at_N_exact'])
plt.title('Runtime vs N-qubit sizes')
plt.savefig('img/Runtime.png')
plt.close()

print('q fidelities_all_runs', q_fidelities_all_runs)
plt.plot(q_fidelities_all_runs)
plt.title('q fidelities at all runs')
plt.savefig('img_torch/q_fidelities.png')

print('TVs_all_runs', TVs_all_runs)
plt.plot(TVs_all_runs)
plt.title('TVs at all runs')
plt.savefig('img_torch/TVs.png')





