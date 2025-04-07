# Author: Aryan Deshwal

import numpy as np
np.random.seed(42) # for fixed behavior
def rand_ising(nodes):
    Q = np.zeros((nodes, nodes))
    n_side = np.sqrt(nodes).astype(np.int32)
    # all the right edges
    for i in range(n_side):
    	for j in range(n_side-1):
            node = i*n_side + j
		    # assign edges weights randomly as in the paper
            par = 4.95*np.random.rand() + 0.05
            if (np.random.rand() > 0.5):
                par = par*-1
            Q[node,node+1] = par
            Q[node+1,node] = Q[node,node+1]
    # all the down edges
    for i in range(n_side-1):
    	for j in range(n_side):
            node = i*n_side + j
            par = 4.95*np.random.rand() + 0.05
            if (np.random.rand() > 0.5):
                par = par*-1         
            Q[node,node+n_side] = par 
            Q[node+n_side,node] = Q[node,node+n_side]
    # assign sign of each edge parameter positive or negative with probability half
    #rand_sign = np.tril((np.random.rand(nodes, nodes) > 0.5)*2-1, -1)
    #rand_sign = rand_sign + rand_sign.T
    #Q = rand_sign * Q
    return Q
def ising_moments(Q):
    nodes = Q.shape[0]
    bin_vals = np.zeros((2**nodes, nodes))
    for i in range(2**nodes):
            bin_vals[i] = np.array(list(np.binary_repr(i).zfill(nodes))).astype(np.int8)
    bin_vals[bin_vals == 0] = -1
    n_vectors = bin_vals.shape[0]
    
    pdf_vals = np.zeros(n_vectors)
    for i in range(n_vectors):
            pdf_vals[i] = np.exp(np.dot(bin_vals[i, :], Q).dot(bin_vals[i, :].T))
    #pdf_vals = np.exp(np.dot(bin_vals, Q).dot(bin_vals.T))
    norm_const = np.sum(pdf_vals)
    ising_moments = np.zeros((nodes, nodes))
    # Second moment for each pair of values
    for i in range(nodes):
           for j in range(nodes):
                bin_pair = bin_vals[:, i]*bin_vals[:, j]
                ising_moments[i][j] = np.sum(bin_pair*pdf_vals)/norm_const
    return ising_moments

def KL_divergence(Q, moments, x):
    if (x.ndim == 2):
        x = x.reshape(x.shape[1])
    Theta_P = Q
    nodes = Q.shape[0]
    bin_vals = np.zeros((2**nodes, nodes))
    for i in range(2**nodes):
            bin_vals[i] = np.array(list(np.binary_repr(i).zfill(nodes))).astype(np.int8)
    bin_vals[bin_vals == 0] = -1
    n_vectors = bin_vals.shape[0]
    P_vals = np.zeros(n_vectors)
    for i in range(n_vectors):
            P_vals[i] = np.exp(np.dot(bin_vals[i, :], Theta_P).dot(bin_vals[i, :].T))
    Zp = np.sum(P_vals)  # partition function of random variable 
    Theta_Q = np.tril(Theta_P, -1)
    nnz_Q = Theta_Q!=0
    Theta_Q[nnz_Q] = Theta_Q[nnz_Q]*x.T
    Theta_Q = Theta_Q + Theta_Q.T
    Q_vals = np.zeros(n_vectors)
    for i in range(n_vectors):
        Q_vals[i] = np.exp(np.dot(bin_vals[i, :], Theta_Q).dot(bin_vals[i, :].T))
    Zq = np.sum(Q_vals) # partition function of random variable Q
    KL = np.sum(np.sum((Theta_P - Theta_Q)*moments)) + np.log(Zq) - np.log(Zp)
    if (KL < 0):
        print("KL less than zero!")
        print(x)
        print(np.sum(Q<0)/2)
        print(np.sum(np.sum((Theta_P - Theta_Q))))
        print(np.log(Zq))
        print(np.log(Zp))
    return KL



def model_run(x, nodes, Q, im):
    # Q = rand_ising(nodes)
    # im = ising_moments(Q)
    return KL_divergence(Q, im, x)

#print(rand_ising(9))
if __name__ == '__main__':
    Q = rand_ising(9)
    mm = ising_moments(Q)
    for i in range(2**12):
            x = np.array(list(np.binary_repr(i).zfill(12))).astype(np.int8)
            print(KL_divergence(Q, mm, x))
