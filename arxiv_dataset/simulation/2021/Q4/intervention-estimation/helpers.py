"""
utility and helper functions
"""

import numpy as np
import numpy.linalg as LA
import networkx as nx
from networkx.algorithms.clique import find_cliques as find_maximal_cliques
import itertools

def get_precision_cov(B,Omega):
    'Given B and Omega where noise is ~ N(0,Omega), return precision matrix'
    # B: autoregressive matrix
    # Sigma: diagonal noise matrix
    p = len(B)
    Theta = (np.eye(p)-B)@LA.inv(Omega)@(np.eye(p)-B).T
    Cov = LA.inv(Theta)
    #Cov = LA.inv(np.eye(p)-B)@Omega@(LA.inv(np.eye(p)-B)).T
    return Theta, Cov

def return_cliques(C):
    'Given matrix C, return the cliques in supp(C)'
    D = np.abs(C) > 1e-10
    # get non-zero diagonal
    S = np.where(np.diag(D))[0]
    G = nx.from_numpy_matrix(D)
    cliques = list(find_maximal_cliques(G))
    single_nodes = []
    for clique in cliques:
        if len(clique) == 1:
            if np.isin(clique,S)[0] == False:
                single_nodes.append(clique)

    for single_node in single_nodes:
        cliques.remove(single_node)
        
    return cliques    


def create_intervention(p,I_size,density,mu=0,shift=1.0,plus_variance=0.5,variance=1.0,\
                              tol=1e-6,B_distortion_amplitude=0.0,perfect_intervention=False):
    '''
    enable shift intervention, changing variance of noise interventions, and perfect interventions for one DAG
    
    internal noise components for intervened nodes:     N(mu+shift,variance+plus_variance)
                              for non-intervened nodes: N(mu,variance)  
                              
    if perfect_intervention is true, make B^2_{pa(i),i} = 0 in second DAG. 
    otherwise distort them by B_distortion_amplitude
    '''
    # create base B
    B = np.random.uniform(-1,-0.25,[p,p])* np.random.choice([-1,1],size=[p,p])
    B = np.triu(B)
    np.fill_diagonal(B,0)
    edge_indices = np.triu(np.random.uniform(size=[p,p])<(density/p))
    B = B*edge_indices    
    #G = nx.to_networkx_graph(B,create_using=nx.DiGraph)
    # Intervention set
    I = list(np.sort(np.random.choice(p,I_size,replace=False)))

    # internal noises will be N(mean,variance) for G1, N(mean+intervention_side*shift,variance) for G2
    mu1 = mu*np.ones(p)
    mu2 = mu*np.ones(p)
    variance1 = variance*np.ones(p)
    variance2 = variance*np.ones(p)
    # shift the noise means
    mu2[I] += shift
    # increase the noise variances
    variance2[I] += plus_variance
    Omega1 = mu1**2 + variance1
    Omega2 = mu2**2 + variance2

    B1 = B.copy(); B2 = B.copy()    
    
    if perfect_intervention is False:
        # apply imperfect changes to B, or keep it the same if  B_distortion_amplitude=0.0
        B_distortion = np.sign(B2) * B_distortion_amplitude
        B_distortion[:,np.delete(np.arange(p),I)] = 0
        B2 -= B_distortion
    elif perfect_intervention is True:
        # apply perfect interventions, remove all parents to child connections for target nodes
        B2[:,I] = 0
    

    # now ready to get precision (or generalized precision) matrix
    Theta1, Cov1 = get_precision_cov(B1, np.diag(Omega1))
    Theta2, Cov2 = get_precision_cov(B2, np.diag(Omega2))
    Delta_Theta = np.abs(Theta1-Theta2)>tol
    S_Delta = np.where(np.diag(Delta_Theta))[0]   
    
    G1 = nx.to_networkx_graph(B1,create_using=nx.DiGraph)
    G2 = nx.to_networkx_graph(B2,create_using=nx.DiGraph)
    
    return B1,G1,mu1,variance1,Omega1,Theta1,Cov1,B2,G2,mu2,variance2,Omega2,Theta2,Cov2,Delta_Theta,S_Delta,I


def create_multiple_intervention(p,I_size,n_interventions,density,mu=0,shift=1.0,plus_variance=0.5,variance=1.0,\
                              tol=1e-6,B_distortion_amplitude=0.0,perfect_intervention=False):
    '''
    enable shift intervention, changing variance of noise interventions, and perfect interventions for one DAG
    
    internal noise components for intervened nodes:     N(mu+shift,variance+plus_variance)
                              for non-intervened nodes: N(mu,variance)  
                              
    if perfect_intervention is true, make B^2_{pa(i),i} = 0 in second DAG. 
    otherwise distort them by B_distortion_amplitude

    generate multiple such interventional settings.
    '''
    # create base B
    B = np.random.uniform(-1,-0.25,[p,p])* np.random.choice([-1,1],size=[p,p])
    B = np.triu(B)
    np.fill_diagonal(B,0)
    edge_indices = np.triu(np.random.uniform(size=[p,p])<(density/p))
    B = B*edge_indices    

    'observational case will be setting_0'
    mu1 = mu*np.ones(p)
    variance1 = variance*np.ones(p)
    Omega1 = mu1**2 + variance1
    B1 = B.copy()
    Theta1, Cov1 = get_precision_cov(B1, np.diag(Omega1))
    # adjacency matrix
    dag = B.copy(); dag[np.where(dag)] = 1
    skeleton = dag + dag.T
    
    setting_list = {}
    setting_list['setting_0']= {}
    setting_list['setting_0']['B'] = B1
    setting_list['setting_0']['mu'] = mu1
    setting_list['setting_0']['variance'] = variance1
    setting_list['setting_0']['Omega'] = Omega1
    setting_list['setting_0']['Theta'] = Theta1
    setting_list['setting_0']['Cov'] = Cov1
    setting_list['setting_0']['dag'] = dag
    setting_list['setting_0']['skeleton'] = skeleton

    I_all = {}
    I_parents_all = {}

    'randomly choosing I_size target nodes without replacement n_interventions times'
    random_ordering = np.random.permutation(p)
    for idx_intervention in range(1,n_interventions+1):
        setting_list['setting_%d'%idx_intervention] = {}
        I = sorted(random_ordering[(idx_intervention-1)*I_size:(idx_intervention)*I_size])
        I_all['setting_%d'%idx_intervention] = I
        I_parents_all['setting_%d'%idx_intervention] = [list(np.where(B[:,i])[0]) for i in I]

        setting_list['setting_%d'%idx_intervention]['I'] = I
        # internal noises will be N(mean,variance) for G1, N(mean+intervention_side*shift,variance) for G2
        mu2 = mu*np.ones(p)
        variance2 = variance*np.ones(p)
        # shift the noise means
        mu2[I] += shift
        # increase the noise variances
        variance2[I] += plus_variance
        Omega2 = mu2**2 + variance2
        B2 = B.copy()        
        if perfect_intervention is False:
            # apply imperfect changes to B, or keep it the same if  B_distortion_amplitude=0.0
            B_distortion = np.sign(B2) * B_distortion_amplitude
            B_distortion[:,np.delete(np.arange(p),I)] = 0
            B2 -= B_distortion
        elif perfect_intervention is True:
            # apply perfect interventions, remove all parents to child connections for target nodes
            B2[:,I] = 0
    
        # now ready to get precision (or generalized precision) matrix
        Theta2, Cov2 = get_precision_cov(B2, np.diag(Omega2))
        Delta_Theta = np.abs(Theta1-Theta2)>tol
        S_Delta = np.where(np.diag(Delta_Theta))[0]   
        setting_list['setting_%d'%idx_intervention]['B'] = B2
        setting_list['setting_%d'%idx_intervention]['mu'] = mu2
        setting_list['setting_%d'%idx_intervention]['variance'] = variance2
        setting_list['setting_%d'%idx_intervention]['Omega'] = Omega2
        setting_list['setting_%d'%idx_intervention]['Theta'] = Theta2
        setting_list['setting_%d'%idx_intervention]['Cov'] = Cov2
        setting_list['setting_%d'%idx_intervention]['S_Delta'] = S_Delta
        
    return setting_list, I_all, I_parents_all

def sample(B,means,variances,n_samples):
    '''
    Parameters
    ----------
    B : 
        autoregression weight matrix. assumed to be strictly upper triangular
    means : 
        internal noise means.
    variances : 
        internal noise variances.
    n_samples : integer
        number of samples to generate

    Returns
    -------
    samples : n_samples x p matrix
        DAG samples

    '''
    # assume that nodes are given in topological order
    p = len(means)
    samples = np.zeros((n_samples,p))
    noise = np.zeros((n_samples,p))
    for ix, (mean,var) in enumerate(zip(means,variances)):
        noise[:,ix] = np.random.normal(loc=mean,scale=var ** .5, size=n_samples)
        
    for node in range(p):
        parents_node = np.where(B[:,node])[0]
        if len(parents_node)!=0:
            parents_vals = samples[:,parents_node]
            samples[:,node] = np.sum(parents_vals * B[parents_node,node],axis=1) + noise[:,node]
        else:
            samples[:,node] = noise[:,node]
            
    return samples


def counter(I,I_hat,I_parents,I_hat_parents):
    '''
    returns the accuracy stats of I estimation
    
    Parameters
    ----------
    I : list
        Ground truth intervention target set.
    I_hat : list
        estimated intervention target set.
    I_parents : list of lists
        ground truth parents of intervened nodes.
    I_hat_parents : list of lists
        estimated parents of intervened nodes.

    Returns
    -------
    tp_i_num: int
        number of true positives for I estimation.
    fp_i_num: int
        number of false positives for I estimation.
    fn_i_num: int
        number of false negatives for I estimation.
    tp_e : int
        number of true positives for I_parents estimation.
    fp_e : int
        number of false positives for I_parents estimation.
    fn_e : int
        number of false negatives for I_parents estimation.

    '''
    tp_i = list(np.intersect1d(I, I_hat))
    fp_i = list(np.setdiff1d(I_hat,I))
    fn_i = list(np.setdiff1d(I,I_hat))                
    tp_e = 0; fp_e = 0; fn_e = 0; 
    
    I_cup = sorted(list(set(list(I)+list(I_hat))))
    
    for i_idx in range(len(I_cup)):
        if I_cup[i_idx] in fp_i:
            fp_e += len(I_hat_parents[list(I_hat).index(I_cup[i_idx])])
    
        elif I_cup[i_idx] in fn_i:
            fn_e += len(I_parents[list(I).index(I_cup[i_idx])])
            
        elif I_cup[i_idx] in tp_i:
            true_parents = I_parents[list(I).index(I_cup[i_idx])]
            est_parents =  I_hat_parents[list(I_hat).index(I_cup[i_idx])]
            fp_e += len(np.setdiff1d(est_parents,true_parents))
            fn_e += len(np.setdiff1d(true_parents,est_parents))
            tp_e += len(np.intersect1d(true_parents,est_parents))

    tp_i_num = len(tp_i); fp_i_num = len(fp_i); fn_i_num = len(fn_i)
    return tp_i_num, fp_i_num, fn_i_num, tp_e, fp_e, fn_e

def scores(I_tp,I_fp,I_fn,e_tp,e_fp,e_fn):
    I_precision = np.sum(I_tp,0) / (np.sum(I_tp,0)+np.sum(I_fp,0))
    I_recall =  np.sum(I_tp,0) / (np.sum(I_tp,0)+np.sum(I_fn,0))
    I_f1 = np.sum(I_tp,0) / (np.sum(I_tp,0)+(np.sum(I_fp,0)+np.sum(I_fn,0))/2)
    
    e_precision = np.sum(e_tp,0) / (np.sum(e_tp,0)+np.sum(e_fp,0))
    e_recall =  np.sum(e_tp,0) / (np.sum(e_tp,0)+np.sum(e_fn,0))
    e_f1 = np.sum(e_tp,0) / (np.sum(e_tp,0)+(np.sum(e_fp,0)+np.sum(e_fn,0))/2)
    
    return I_precision, I_recall, I_f1, e_precision, e_recall ,e_f1


def find_cpdag_from_dag(A):
    '''
    takes adjacency matrix for a DAG. 
    return CPDAG, and other stuff
    '''
    S, v_structures, directed_edges, undirected_edges = find_v_structures(A)
    S, v_structures, directed_edges, undirected_edges = \
        apply_meek(S, v_structures, directed_edges, undirected_edges)
    
    return S, v_structures, directed_edges, undirected_edges

def analyze_cpdag(S):
    '''
    takes a CPDAG, can be output of an observational algorithm
    return the v_structures, and other stuff of it. just a helper
    '''
    p = S.shape[0]
    v_structures = []
    undirected_loc = np.where((S==1)&(S.T==1)) 
    undirected_edges = list(zip(undirected_loc[0],undirected_loc[1]))
    directed_loc = np.where((S==1)&(S.T==0)) 
    directed_edges = list(zip(directed_loc[0],directed_loc[1]))
    
    for b in range(p):
        # find directed edges incident on b        
        b_dir_adjacent = [b_dir_edge[0] for b_dir_edge in directed_edges if b_dir_edge[1]==b]
        directed_parent_pairs = list(itertools.combinations(b_dir_adjacent, 2))
        for directed_parent_pair in directed_parent_pairs:
            if (S[directed_parent_pair[0],directed_parent_pair[1]] == 0) and \
                (S[directed_parent_pair[1],directed_parent_pair[0]] == 0):
                # for this a,c parent, a-b-c is an unshielded collider
                v_structures.append((directed_parent_pair[0],b,directed_parent_pair[1]))
    
    return S, v_structures, directed_edges, undirected_edges

def find_v_structures(A):
    '''
    takes ground truth adjancecy matrix A
    return the partially directed graph S, with v_structures, directed and undirected edges.
    '''
    p = A.shape[0]
    v_structures = []
    
    # undirected skeleton
    S = A.copy()
    S[np.where(A.T)] = 1 
    
    for i in range(p):
        pa_i = np.where(A[:,i])[0]
        if len(pa_i) < 2:
            # no v-structure incident on iA
            continue
        else:
            parent_pairs = list(itertools.combinations(pa_i, 2))
            for parent_pair in parent_pairs:
                if S[parent_pair[0],parent_pair[1]] == 0:
                    # for this j,k parents, j-i-k is an unshielded collider, i.e., v-structure
                    v_structures.append((parent_pair[0],i,parent_pair[1]))
 
    'now create partially directed graph'
    for v in v_structures:
        # remove the wrong side of the arrow
        S[v[1],v[0]] = 0
        S[v[1],v[2]] = 0            

    undirected_loc = np.where((S==1)&(S.T==1)) 
    undirected_edges = list(zip(undirected_loc[0],undirected_loc[1]))
    directed_loc = np.where((S==1)&(S.T==0)) 
    directed_edges = list(zip(directed_loc[0],directed_loc[1]))

    return S, v_structures, directed_edges, undirected_edges
            

def apply_meek(S,v_structures,directed_edges,undirected_edges):
    '''
    takes skeleton S and its v-structures, directed and undirected edges.
    apply meek rules to orient as many edges as possible.
    final partially directed S is the essential graph
    '''
    p = S.shape[0]
    overall_flag = True
    while overall_flag is True:
        overall_flag = False
        flag = True
        while flag is True:
            ' meek rule 1 '
            # if a to b is oriented. b-c is not. and a and c are not adjacent. orient b to c
            flag = False
            for d_edge in directed_edges:
                a,b = d_edge
                # find undirected edges to b
                b_undir_adjacent = [b_undir_edge[0] for b_undir_edge in undirected_edges if b_undir_edge[1]==b]
                for c in b_undir_adjacent:
                    if (S[a,c] == 0) and (S[c,a] == 0):
                        # since a-b-c is not a v-structure, we must have b to c
                        S[b,c] = 1; S[c,b] = 0
                        directed_edges.append((b,c))
                        undirected_edges.remove((b,c))
                        undirected_edges.remove((c,b))
                        flag = True
                        overall_flag = True
        
        flag = True
        while flag is True:
            ' meek rule 2 '
            # a to b, c to b are oriented. d-a, d-c, d-b are not. then orient d to b
            flag = False
            for b in range(p):
                # directed edges to b
                b_dir_adjacent = [b_dir_edge[0] for b_dir_edge in directed_edges if b_dir_edge[1]==b]
                # undirected edges to b
                b_undir_adjacent = [b_undir_edge[0] for b_undir_edge in undirected_edges if b_undir_edge[1]==b]
                for d in b_undir_adjacent:
                    d_undir_adjacent = [d_undir_edge[0] for d_undir_edge in undirected_edges if d_undir_edge[1]==d]
                    # common undirected adjancents to b and d, so these will be possible a and c's
                    bd_undir_adjacent = np.intersect1d(b_dir_adjacent,d_undir_adjacent)
                    # if such a-c's exist, then orient d to b
                    if len(bd_undir_adjacent) > 0:
                        S[d,b] = 1; S[b,d] = 0
                        directed_edges.append((d,b))
                        undirected_edges.remove((d,b))
                        undirected_edges.remove((b,d))
                        flag = True
                        overall_flag = True
        
        flag = True
        while flag is True:         
            ' meek rule 3 '
            # a to b, b to c oriented.  a-c is not. then orient a to c
            flag = False
            for b in range(p):
                # directed edges incident on b, possible a nodes
                a_list = [dir_edge[0] for dir_edge in directed_edges if dir_edge[1]==b]
                # directed edges from b to c
                c_list = [dir_edge[1] for dir_edge in directed_edges if dir_edge[0]==b]
                for a in a_list:
                    for c in c_list:
                        if (a,c) in undirected_edges:
                            # orient a to c
                            S[a,c] = 1; S[c,a] = 0
                            directed_edges.append((a,c))
                            undirected_edges.remove((a,c))
                            undirected_edges.remove((c,a))                
                            flag = True
                            overall_flag = True
        
        
        flag = True
        while flag is True:
            ' meek rule 4 '
            # a to b, d to a are oriented. d-c, a-c, b-c are not. then orient c to b
            flag = False
            for a in range(p):
                # directed edges incident on a, possible d nodes
                d_list = [dir_edge[0] for dir_edge in directed_edges if dir_edge[1]==a]    
                # directed edges from a to b, possible b nodes
                b_list = [dir_edge[1] for dir_edge in directed_edges if dir_edge[0]==a]
                # undirected adjancents to a, possible c nodes
                c_list = [undir_edge[0] for undir_edge in undirected_edges if undir_edge[1]==a]
                for c in c_list:
                    c_undir_adjacent = [c_undir_edge[0] for c_undir_edge in undirected_edges if c_undir_edge[1]==c]
                    if (len(np.intersect1d(c_undir_adjacent,d_list)) > 0 and len(np.intersect1d(c_undir_adjacent,b_list)) > 0):
                        # such b c d triple exist. orient c to b.
                        S[c,b] = 1; S[b,c] = 0
                        directed_edges.append((c,b))
                        undirected_edges.remove((c,b))
                        undirected_edges.remove((b,c))
                        flag = True
                        overall_flag = True
                    
    return S, v_structures, directed_edges, undirected_edges

def intervention_CPDAG(S,v_structures,I,Ij_parents):
    '''
    takes a CPDAG S and v_structures
    apply the intervention knowledge
    return new I-CPDAG
    '''
    #nonI_nodes = np.delete(np.arange(S.shape[0]),I)
    #undirected_loc = np.where((S==1)&(S.T==1)) 
    #undirected_edges = list(zip(undirected_loc[0],undirected_loc[1]))
    directed_loc = np.where((S==1)&(S.T==0)) 
    directed_edges = list(zip(directed_loc[0],directed_loc[1]))
    
    I_S = S.copy()
    for i_idx in range(len(I)):
        i = I[i_idx]
        pa_i = Ij_parents[i_idx]
        # besides I's and Ij_parents, no possible parents
        #I_S[nonI_nodes,i] = 0        
        I_S[pa_i,i] = 1
        I_S[i,pa_i] = 0
        
    I_undirected_loc = np.where((I_S==1)&(I_S.T==1)) 
    I_undirected_edges = list(zip(I_undirected_loc[0],I_undirected_loc[1]))
    I_directed_loc = np.where((I_S==1)&(I_S.T==0)) 
    I_directed_edges = list(zip(I_directed_loc[0],I_directed_loc[1]))
    I_S, I_v_structures, I_directed_edges, I_undirected_edges = apply_meek(I_S,v_structures,I_directed_edges,I_undirected_edges)  
    I_new_directed_edges = [I_directed_edges[i] for i in range(len(I_directed_edges)) if I_directed_edges[i] not in directed_edges]
    return I_S, I_v_structures, I_directed_edges, I_undirected_edges, I_new_directed_edges


def multiple_intervention_CPDAG(S,v_structures,I_all,I_parents_all):
    '''
    takes a CPDAG S and v_structures
    apply multiple interventional settings
    return final I-CPDAG
    '''    
    #directed_loc = np.where((S==1)&(S.T==0)) 
    #directed_edges = list(zip(directed_loc[0],directed_loc[1]))
    
    I_S = S.copy()
    for idx_setting in range(1,len(I_all)+1):
        I = I_all['setting_%d'%idx_setting]
        I_parents = I_parents_all['setting_%d'%idx_setting]
        Ij_parents = [list(np.setdiff1d(I_parents[i], I)) for i in range(len(I))]

        for i_idx in range(len(I)):
            i = I[i_idx]
            pa_i = Ij_parents[i_idx]
            # besides I's and Ij_parents, no possible parents
            #I_S[nonI_nodes,i] = 0        
            I_S[pa_i,i] = 1
            I_S[i,pa_i] = 0        
            
        I_undirected_loc = np.where((I_S==1)&(I_S.T==1)) 
        I_undirected_edges = list(zip(I_undirected_loc[0],I_undirected_loc[1]))
        I_directed_loc = np.where((I_S==1)&(I_S.T==0)) 
        I_directed_edges = list(zip(I_directed_loc[0],I_directed_loc[1]))
        I_S, v_structures, I_directed_edges, I_undirected_edges = apply_meek(I_S,v_structures,I_directed_edges,I_undirected_edges)  
        #I_new_directed_edges = [I_directed_edges[i] for i in range(len(I_directed_edges)) if I_directed_edges[i] not in directed_edges]
    return I_S
            
        

def SHD_CPDAG(S1,S2):
    '''
    computes SHD for two CPDAGs.
    minimum number of edge additions/deletions/conversions between
    directed and undirected to convert one graph to the other
    '''
    #S1, v_structures_1, directed_edges_1, undirected_edges_1 = analyze_cpdag(S1)
    #S2, v_structures_2, directed_edges_2, undirected_edges_2 = analyze_cpdag(S2)
    
    diff = np.abs(S1-S2)
    # totally differing undirected edges OR flip the direction.
    diff_loc_1 = np.where((diff==1)&(diff.T==1))
    diff_edges_1 = list(zip(diff_loc_1[0],diff_loc_1[1]))
    # other changes
    diff_loc_2 = np.where((diff==1)&(diff.T==0)) 
    diff_edges_2 = list(zip(diff_loc_2[0],diff_loc_2[1]))    
    
    shd = int(len(diff_edges_1)/2 + len(diff_edges_2))
    return shd 
