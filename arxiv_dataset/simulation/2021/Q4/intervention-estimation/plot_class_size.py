"""
Empirical findings for the max. class size (exponential component of our algo's comp. complexity) and 
size of the changed nodes. 
"""

import numpy as np
from helpers import get_precision_cov
import networkx as nx
from networkx.algorithms.clique import find_cliques as find_maximal_cliques
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
import numpy.linalg as LA
from config import SIMULATIONS_FIGURES_FOLDER


def marginal_theta_from_cov(Cov,M):
    'Given covariance matrix, take inverse of Sigma_{M,M}'
    'faster than using marginal_theta_from_theta function '
    ThetaM = LA.inv(Cov[M][:,M])
    return ThetaM

def build_descendants(Cov1,Cov2,M,S,tol=1e-9):
    '''
    given that elements of M is covered, i.e., elements of M has all their 
    Pa(An^I(j)) contained in M, compute pairwise ancestor-descendant relationship
    from M to S
    '''
    M_des = np.zeros([len(M),len(S)])
    for j_idx in range(len(M)):
        for s_idx in range(len(S)):
            Delta_Theta_sj = marginal_theta_from_cov(Cov1, [M[j_idx],S[s_idx]]) \
                            - marginal_theta_from_cov(Cov2, [M[j_idx],S[s_idx]])

            M_des[j_idx,s_idx] = np.abs(Delta_Theta_sj[0,1]) > tol
            
    return M_des

def single_marginal_noise(Theta,Cov,B,Omega,G):
    'notice that for a single node, Theta = 1/sigma_j^2'
    Omega_tilde = []
    M = np.arange(len(B))
    for i in M:
        Omega_tilde.append(1/Cov[i,i])
        #Omega_tilde.append(marginal_noise(Theta,Cov,B,Omega,G,[i]))
        
    return np.squeeze(Omega_tilde)

def create_random_sem(p,I_size,density,Bnoise_amplitude=0.2,tol=1e-6):
    '''
    Create Erdos-Renyi random graphs with p nodes, 
    '''
    B = np.random.uniform(-1,-0.25,[p,p])* np.sign(np.random.uniform(-1,0.25,[p,p]))
    B = np.triu(B)
    np.fill_diagonal(B,0)
    edge_indices = np.triu(np.random.uniform(size=[p,p])<(density/p))
    B = B*edge_indices
    # Intervention set
    I = np.sort(np.random.choice(p,I_size,replace=False))
    # chance noise factors for set I
    diag_Omega1 = np.random.uniform(1.0,1.0,size=(p))
    Omega1 = np.diag(diag_Omega1)
    
    diag_Omega2 = diag_Omega1.copy()
    #diag_Omega2[I] += randomize_sign(np.random.uniform(0.2,0.3,len(I)))
    diag_Omega2[I] += np.random.uniform(0.25,1.0,len(I))
    Omega2 = np.diag(diag_Omega2)
    
    B1 = B.copy()
    B2 = B.copy()
    
    Bnoise = np.zeros(B2.shape)
    Bnoise[:,I] = -Bnoise_amplitude
    Bnoise = np.triu(Bnoise,k=1)* (np.abs(B1)>0)
    B2 = B1 + Bnoise
    
    G1 = nx.to_networkx_graph(B1,create_using=nx.DiGraph)
    G2 = nx.to_networkx_graph(B2,create_using=nx.DiGraph)
    
    # get precision and covariance matrices
    Theta1, Cov1 = get_precision_cov(B1,Omega1)
    Theta2, Cov2 = get_precision_cov(B2,Omega2)
    Delta_Theta = np.abs(Theta1-Theta2)>tol
    S_Delta = np.where(np.diag(Delta_Theta))[0]   

    return B1,Omega1,G1,Theta1,Cov1,B2,Omega2,G2,Theta2,Cov2,Delta_Theta,S_Delta,I

def random_dag(p,I_size,density):
    B = np.random.uniform(-1,-0.25,[p,p])* np.random.choice([-1,1],size=[p,p])
    B = np.triu(B)
    np.fill_diagonal(B,0)
    edge_indices = np.triu(np.random.uniform(size=[p,p])<(density/p))
    B = B*edge_indices    
    #G = nx.to_networkx_graph(B,create_using=nx.DiGraph)
    A = np.zeros(B.shape)
    A[np.where(B)] = 1
    #I = list(np.sort(np.random.choice(p,I_size,replace=False)))
    return A

def find_num_paths(A,q=1):
    k = 1
    sum_Ak = np.zeros(A.shape)
    
    b = matrix_power(A,k)
    while np.sum(b) > 0:
        sum_Ak += b*q**k
        k += 1
        b = matrix_power(A,k)
        
    return sum_Ak

def no_I_ancestor_prob(n_repeat,p,I_size,q):
    res = np.zeros(n_repeat)
    A = random_dag(p,I_size,p)
    N = find_num_paths(A,q)
    for _ in range(n_repeat):
        I = list(np.sort(np.random.choice(p,I_size,replace=False)))
        res[_] = np.prod(1-np.clip(N[I][:,I],0,1).flatten())
        
    return res

def repeat_find_sizes(n_repeat,p,I_size,density):    
    p_delta = np.zeros(n_repeat)
    J0_sizes = np.zeros(n_repeat)
    max_Al_sizes = np.zeros(n_repeat)

    for _ in range(n_repeat):
        B1,Omega1,G1,Theta1,Cov1,B2,Omega2,G2,Theta2,Cov2,Delta_Theta,S_Delta,I \
            = create_random_sem(p=p, I_size=I_size, density=density,Bnoise_amplitude=0.0,tol=1e-9)

        res = algorithm_size(p,I_size,density)
        p_delta[_] = res[0]
        J0_sizes[_] = res[1]
        max_Al_sizes[_] = res[3] 
        
    return p_delta, J0_sizes, max_Al_sizes

def algorithm_size(p,I_size,density,tol=1e-9):
    B1,Omega1,G1,Theta1,Cov1,B2,Omega2,G2,Theta2,Cov2,Delta_Theta,S_Delta,I \
        = create_random_sem(p=p, I_size=I_size, density=density,Bnoise_amplitude=0.0,tol=tol)

    p = len(B1)
    # neutralizing ancestor set for j nodes
    N = [[] for i in range(p)]
    diff_marginal_noise = single_marginal_noise(Theta1,Cov1,B1,Omega1,G1)-single_marginal_noise(Theta2,Cov2,B2,Omega2,G2)
    J0 = np.intersect1d(np.where(np.abs(diff_marginal_noise)<tol)[0],S_Delta)
    for j in J0:
        N[j] = []

    S_d = np.setdiff1d(S_Delta,J0)
    'build J0_ancestor sets'
    J0_anc = build_descendants(Cov1,Cov2,J0,S_d,tol).T
    J0_anc_lists = [J0[np.where(J0_anc[i])[0]] for i in range(len(J0_anc))]
    J0_anc_size = [len(J0_anc_lists[i]) for i in range(len(J0_anc_lists))]

    size_argsort_order = np.argsort(J0_anc_size)
    A_mat = np.zeros([len(J0_anc_size),len(J0_anc_size)])
    
    for i in range(len(J0_anc_size)):
        for j in range(len(J0_anc_size)):
            #A_mat[i,j] = np.array_equal(J0_anc_lists[i],J0_anc_lists[j])
            A_mat[i,j] = np.array_equal(J0_anc_lists[size_argsort_order[i]],J0_anc_lists[size_argsort_order[j]])
    
    #A_groups = list(find_maximal_cliques(nx.from_numpy_matrix(A_mat[size_argsort_order][:,size_argsort_order])))
    A_groups = list(find_maximal_cliques(nx.from_numpy_matrix(A_mat)))
    A_groups = [np.sort([size_argsort_order[i] for i in A]) for A in A_groups]
    # map to the correct indices in graph
    A_groups = [list(S_d[A_groups[i]]) for i in range(len(A_groups))]
    Al_sizes = [len(A) for A in A_groups]
    return len(S_Delta), len(J0), len(I), max(Al_sizes)



#%%
''' for p = 100, see how it changes with density first '''
n_repeat = 1000
p = 100
I_size = 5
#density_list = [1,2,3,4,5,6,7,8,9,10]
density_list = [3,5,10]

p_delta_p100_I5 = np.zeros((len(density_list),n_repeat))
Al_p100_I5 = np.zeros((len(density_list),n_repeat))

for i in range(len(density_list)):
    res_p100_i = repeat_find_sizes(n_repeat=n_repeat,p=p,I_size=I_size,density=density_list[i])
    p_delta_p100_I5[i] = res_p100_i[0]
    Al_p100_I5[i] = res_p100_i[-1]
    print(i)
    
''' for p = 100, see how it changes with density first '''
n_repeat = 1000
p = 100
I_size = 10
#density_list = [1,2,3,4,5,6,7,8,9,10]
density_list = [3,5,10]


p_delta_p100_I10 = np.zeros((len(density_list),n_repeat))
Al_p100_I10 = np.zeros((len(density_list),n_repeat))

for i in range(len(density_list)):
    res_p100_i = repeat_find_sizes(n_repeat=n_repeat,p=p,I_size=I_size,density=density_list[i])
    p_delta_p100_I10[i] = res_p100_i[0]
    Al_p100_I10[i] = res_p100_i[-1]
    print(i)
    

#%%
xticks_size = 12
yticks_size = 12
xlabel_size = 14
ylabel_size = 14
legend_size = 12
legend_loc = 'upper left'
linewidth = 2.5
linestyle = '--'
markersize = 8
markers = ['o','v','P']


#%%
# plot distribution of max class size and S_Delta size.

percentile_loc = (p_delta_p100_I5.shape[-1]*np.arange(0.1,1,0.1)).astype(int)
percentile_ticks = (100*np.arange(0.1,1,0.1)).astype(int)

# for c in pick_densities:
#     h1 = np.histogram(Al_p100_I5[c],bins=10)
#     h2 = np.histogram(p_delta_p100_I5[c],bins=10)
#     plt.plot(h1[1][1:],h1[0],'-o')
#     plt.plot(h2[1][1:],h2[0],'-x')
plt.figure('p=100, |I| = 5')
plt.plot(percentile_ticks,np.sort(Al_p100_I5[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I5[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I5[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.plot(percentile_ticks,np.sort(p_delta_p100_I5[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I5[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I5[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.ylim([0,37])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Percentile',size=xlabel_size)

plt.grid()
plt.legend(['max $|A_l|$ w/ c=3','max $|A_l|$ w/ c=5','max $|A_l|$ w/ c=10','$|S_\Delta|$ w/ c=3', '$|S_\Delta|$ w/ c=5','$|S_\Delta|$ w/ c=10'],loc=legend_loc)
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/revised_p100_I5.eps')


plt.figure('p=100, |I| = 10')
plt.plot(percentile_ticks,np.sort(Al_p100_I10[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I10[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(Al_p100_I10[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.plot(percentile_ticks,np.sort(p_delta_p100_I10[0][percentile_loc]),marker=markers[0],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I10[1][percentile_loc]),marker=markers[1],linestyle=linestyle,linewidth=linewidth,markersize=markersize)
plt.plot(percentile_ticks,np.sort(p_delta_p100_I10[2][percentile_loc]),marker=markers[2],linestyle=linestyle,linewidth=linewidth,markersize=markersize)

plt.ylim([0,70])
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Percentile',size=xlabel_size)

plt.grid()
plt.legend(['max $|A_l|$ w/ c=3','max $|A_l|$ w/ c=5','max $|A_l|$ w/ c=10','$|S_\Delta|$ w/ c=3', '$|S_\Delta|$ w/ c=5','$|S_\Delta|$ w/ c=10'],loc=legend_loc)
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/revised_p100_I10.eps')


#%%
plt.figure('p=100, |I| = 10')

plt.plot(np.sort(p_delta_p100_I10[0]),'-ro',markersize=0.5)
plt.plot(np.sort(p_delta_p100_I10[1]),'-bo',markersize=0.5)
plt.plot(np.sort(p_delta_p100_I10[2]),'-go',markersize=0.5)

plt.plot(np.sort(Al_p100_I10[0]),'-ro',markersize=0.5)
plt.plot(np.sort(Al_p100_I10[1]),'-bo',markersize=0.5)
plt.plot(np.sort(Al_p100_I10[2]),'-go',markersize=0.5)
plt.grid()
plt.legend(['c=3','c=5','c=10'])
#plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/revised_p100_I10.eps')

