import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from multiprocessing import Pool

def update_p(p0, CR_l, Ql, CP_l):
    for i in range(nbinl):
        CR_l[i] = np.dot(np.dot(p0.T.I, CP_l[i]), p0.I)
        Ql[i] = np.dot(CP_l[i], p0.I)
        Ql_all = np.mat(Ql.sum(axis=0))
        CR_l_all = np.mat(CR_l.sum(axis=0))
        p1_T = np.abs(np.dot(Ql_all, CR_l_all.I))
        vj = np.zeros(number_bin)
        for j in range(number_bin):
            vj[j] = p1_T[j, :].sum()
        epsilon = 1e-10
        p2_T = np.zeros_like(p0)
        for j in range(number_bin):
            p2_T[:, j] = (p1_T[:, j]+epsilon) / (vj[j]+epsilon) 
        p0 = p2_T.T
    return p0, CR_l

def a1_iter(seed):
    np.random.seed(seed)
    times = 0
    n_times = len(n_a1[seed])
    J_all = np.zeros(n_times)
    p0_all = np.zeros((n_times, number_bin, number_bin))

    CP_l = np.zeros((nbinl, number_bin, number_bin))
    CP_l[:] = gg_ps[:]
    while(times < n_times):
        n_iter = 0
        CR_l = np.zeros_like(CP_l)
        Ql = np.zeros_like(CP_l)
        p_random = np.random.rand(number_bin, number_bin)
        for i in range(number_bin):
            p_random[i, i] = 2*p_random[i, :].sum()
            p_random[:, i] = p_random[:,i] / p_random[:, i].sum()
        p0 = np.mat(p_random)
        JJ = []
        pp0 = []
        while(n_iter < A1_max_iter):
            p0, CR_l = update_p(p0, CR_l, Ql,CP_l)
            J = 0
            diag_CR = np.zeros(number_bin)
            for i in range(nbinl):
                diag_CR[:] = np.diagonal(CR_l[i])
                CR_l[i] = 0
                for j in range(number_bin):
                    CR_l[i,j,j] = abs(diag_CR[j])
                J += 0.5*np.linalg.norm(CP_l[i] - np.dot(np.dot(p0.T, CR_l[i]), p0))**2
            JJ.append(J)
            pp0.append(p0)
            if n_iter >= 3:
                judge_J = (JJ[n_iter-1] + JJ[n_iter-2] + JJ[n_iter-3])/3*1.5
                if JJ[n_iter] >= judge_J:
                    break
            if n_iter >= 1000:
                if np.argmin(JJ) < 0.80*n_iter:
                    break
            n_iter += 1
        J_all[times] = JJ[np.argmin(JJ)]
        p0_all[times] = pp0[np.argmin(JJ)]
        times += 1
    return J_all, p0_all

def update_CR_l(W, Vl, nbinl, number_bin):
    U = np.zeros((number_bin*number_bin, number_bin))
    Vec_Vl = np.zeros((nbinl, number_bin*number_bin, 1))
    CL = np.zeros((nbinl, number_bin, number_bin))
    for i in range(number_bin):
        U[:,i] = np.kron(W[:, i], W[:, i])
    for i in range(nbinl):
        Vec_Vl[i] = Vl[i, :, :].reshape((-1, 1), order='F')
    for i in range(nbinl):
        for j in range(number_bin):
            CL[i,j,j] = abs(np.dot(np.mat(U).I, Vec_Vl[i])[j, 0])
    return CL

def A_plus(A):
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            if(A[i, j] < 0):
                A[i, j] = 0
    return A

def A_minus(A):
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            if(A[i, j] < 0):
                A[i, j] = abs(A[i, j])
            else:
                A[i, j] = 0
    return A

def update_W_NON(V, W, H, Vl, nbinl, number_bin):
    delta1 = np.zeros((number_bin, number_bin))
    delta2 = np.zeros((number_bin, number_bin))
    A = np.zeros((number_bin, number_bin))
    B = np.zeros((number_bin, number_bin))
    W1 = np.zeros_like(W)
    for n in range(nbinl):
        delta1[:] += A_plus(np.dot(np.multiply(V[n], Vl[n]), H[n].T))
        delta2[:] += np.dot(np.multiply(V[n], np.dot(W, H[n])), H[n].T) + A_minus(np.dot(np.multiply(V[n], Vl[n]), H[n].T))
    for i in range(number_bin):
        for j in range(number_bin):
            for b in range(number_bin):
                A[i,j] += W[i,b] / delta2[i,b]
                B[i,j] += W[i,b] * delta1[i,b] / delta2[i,b]
    for i in range(number_bin):
        for j in range(number_bin):
            W1[i,j] = W[i,j] * (delta1[i,j]*A[i,j]+1-B[i,j]) / (delta2[i,j]*A[i,j])
            if W1[i,j] < 0:
                W1[i,j] = W[i,j] * (delta1[i,j]*A[i,j]+1) / (delta2[i,j]*A[i,j]+B[i,j])
    return W1

def a2_iter(nl):
    P_all_test1 = []
    CR_all_test1 = []
    min_J_all = []
    for ip0 in range(len(n_a2[nl])):
        ip = int(n_a2[nl][ip0])

        p0 = np.array(p0_all_sort[ip])

        H = np.zeros((nbinl, number_bin, number_bin))
        Vl = np.zeros((nbinl, number_bin, number_bin))


        Vl[:] = gg_ps[:]

        J_all = []
        p0_all = []
        CR_all = []

        W = p0.T
        judge_W = 1
        n_iter = 0
        CL = update_CR_l(W, Vl, nbinl, number_bin)
        while(n_iter < A2_max_iter and judge_W > 0):
            H[:] = np.dot(CL[:], W.T)
            W1 = update_W_NON(V, W, H, Vl, nbinl, number_bin)
            CL = update_CR_l(W1, Vl, nbinl, number_bin)    #repeat 3
            judge_W = 0
            for i in range(number_bin):
                for j in range(number_bin):
                    if(abs(W1[i, j]-W[i, j]) >= 1e-8):
                        judge_W += 1
            J = 0
            for i in range(nbinl):
                J += 0.5*np.linalg.norm(np.sqrt(V[i])*(Vl[i]-np.dot(np.dot(W1,CL[i]),W1.T)))**2
            J_all.append(J)
            p0_all.append(W1.T)
            CR_all.append(CL)
            W = W1
            if n_iter >= 3:
                judge_J = (J_all[n_iter-1] + J_all[n_iter-2] + J_all[n_iter-3])/3*1.5
                if J_all[n_iter] >= judge_J:
                    break
            if n_iter >= 1000:
                if np.argmin(J_all) < 0.80*n_iter or abs(J_all[n_iter]-J_all[n_iter-1000])/J_all[n_iter]<1e-4:
                    break
            n_iter += 1
        P_all_test1.append(p0_all[np.argmin(J_all)])
        CR_all_test1.append(CR_all[np.argmin(J_all)])
        min_J_all.append(min(J_all))
    return min_J_all, P_all_test1, CR_all_test1

if __name__ == '__main__':
    ################### Below are the input and parameters ####################
    #gg_ps_all is the input power spectra between photo-z bins with shape (n_ps, nbinl, number_bin, number_bin),
    #n_ps is the number of perturbed power spectra, nbinl is the number of \ell bin, number_bin is the number of redshift bin.
    gg_ps_all = np.load('input_powerspectra.npy')

    n_ps, nbinl, number_bin, number_bin = gg_ps_all.shape

    #V is the input weight with shape (nbinl, number_bin, number_bin) and a common choice is the inverse variance matrix of each matrix element 1/(sigma^2).
    sigma_ps = np.load('sigma.npy')
    V = 1/(sigma_ps*sigma_ps)

    # Number of parallel threads, default 40.
    n_threads = 40

    # Number of initial matrices for Algorithm 1, default 1000.
    sum_times = 1000

    # Number of matrices input to algorithm 2 from the output of algorithm 1 (<=sum_times), default 100.
    n_matrix = 100

    # Maximum number of iterations of Algorithm 1, default 1e4.
    A1_max_iter = 1e4

    # Maximum number of iterations of Algorithm 2, default 1e4.
    A2_max_iter = 1e4

    ########## The algorithm consists of Algorithm 1 (FP) and Algorithm 2 (NMF) ##########
    #Algorithm 1
    npool = n_threads
    J_all_sort_all = np.zeros((n_ps, sum_times))
    p0_all_sort_all = np.zeros((n_ps, sum_times, number_bin, number_bin))
    for nr in range(n_ps):
        gg_ps = gg_ps_all[nr]

        n_a1 = np.array_split(range(int(sum_times)), npool)
        pool = Pool(npool)
        rel = pool.map(a1_iter, range(npool))
        pool.close()
        pool.join()
        rel1 = []
        rel2 = []
        for i in range(npool):
            for j in range(len(rel[i][0])):
                rel1.append(rel[i][0][j])
            for j in range(len(rel[i][1])):
                rel2.append(rel[i][1][j])
        J_all = np.array(rel1).reshape(-1,)
        p0_all = np.array(rel2).reshape(-1, number_bin, number_bin)

        J_all_sort_all[nr] = np.copy(J_all[J_all.argsort()])
        p0_all_sort_all[nr] = np.copy(p0_all[J_all.argsort()])

#     # output objective function J (Algorithm 1) with shape (n_ps,sum_times)
#     np.save('J_a1.npy', J_all_sort_all)
#     # output the scattering matrices (Algorithm 1) with shape (n_ps,sum_times,number_bin,number_bin)
#     np.save('P_recon_a1.npy', p0_all_sort_all)

    #Algorithm 2
    npool = n_threads
    J_results = np.zeros((n_ps, n_matrix))
    p_results = np.zeros((n_ps, n_matrix, number_bin, number_bin))
    CR_results = np.zeros((n_ps, n_matrix, nbinl, number_bin, number_bin))
    for nr in range(n_ps):
        gg_ps = gg_ps_all[nr]
        p0_all_sort = p0_all_sort_all[nr]

        n_a2 = np.array_split(range(n_matrix), npool)
        pool = Pool(npool)
        rel = pool.map(a2_iter, range(npool))
        pool.close()
        pool.join()
        rel1 = []
        rel2 = []
        rel3 = []
        for i in range(npool):
            for j in range(len(rel[i][0])):
                rel1.append(rel[i][0][j])
            for j in range(len(rel[i][1])):
                rel2.append(rel[i][1][j])
            for j in range(len(rel[i][2])):
                rel3.append(rel[i][2][j])
        J_all_min = np.array(rel1).reshape(-1,)
        p0_all_min = np.array(rel2).reshape(-1, number_bin, number_bin)
        CR_all_min = np.array(rel3).reshape(-1, nbinl, number_bin, number_bin)

        J_results[nr] = J_all_min
        p_results[nr] = p0_all_min
        CR_results[nr] = CR_all_min

#     # output objective function J (Algorithm 2) with shape (n_ps,n_matrix)
#     np.save('J.npy', J_results)
#     # output the scattering matrices (Algorithm 2) with shape (n_ps,n_matrix,number_bin,number_bin)
#     np.save('P_recon.npy', p_results)
#     # output power spectra between true-z bins (Algorithm 2) with shape (n_ps,n_matrix,nbinl,number_bin,number_bin)
#     np.save('CR_recon.npy', CR_results)

    #save all output (Algorithm 2) into a single file in compressed .npz format.
    np.savez_compressed('recon.npz', J=J_results, P_recon=p_results, CR_recon=CR_results)

