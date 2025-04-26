"""
implement Ghoshal19 algorithm. 
It completely depends on invariant noise variance, let's see how it performs
"""
import numpy as np
import pickle
from config import SIMULATIONS_ESTIMATED_FOLDER, SIMULATIONS_FIGURES_FOLDER
import time
from helpers import create_intervention, sample, counter
from matplotlib import pyplot as plt
from functions import Delta_Theta_func, algorithm_sample
flatten_list = lambda t: list(set([item for sublist in t for item in sublist]))

def ghoshal_algo(S1,S2,lambda_l1=0.1,rho=1.0,th1=5e-3,th2=1e-2,n_max_iter=500,stop_cond=1e-6):
    t0 = time.time()
    p = len(S1)
    Delta_hat, obj_hist = Delta_Theta_func(S1,S2,lambda_l1,rho,n_max_iter,stop_cond,verbose=False)     
    Delta_hat = np.abs(Delta_hat) > th1
    
    # eliminate the invariant vertices
    U = list(np.where(np.sum(Delta_hat,0)==0)[0])
    V = list(np.delete(np.arange(p),U))
    # will be used later
    Delta_hat_V, obj_hist_V = Delta_Theta_func(S1[V][:,V],S2[V][:,V],lambda_l1,rho,n_max_iter,stop_cond,verbose=False)
    
    Vo = V.copy()
    
    # ComputeOrder part
    Order = []
    while len(Vo) > 1:
        Delta_hat_Vo, obj_hist_Vo = Delta_Theta_func(S1[Vo][:,Vo],S2[Vo][:,Vo],lambda_l1,rho,n_max_iter,stop_cond,verbose=False)
        Delta_hat_Vo = np.abs(Delta_hat_Vo) > th1
        S = list(np.where(np.diag(Delta_hat_Vo)==0)[0])
        S_mapped = list(np.asarray(Vo)[S])
        # if nothing has changed, just attach the remaining V as a group
        if len(S_mapped) == 0:
            Order.append(Vo)
            Vo = []
        else:
            Order.append(S_mapped)
            Vo = list(np.delete(np.asarray(Vo),S))    
            
    # OrientEdges part
    Delta = []
    for S in Order:
        for i in S:
            # neighbors
            N_i = np.asarray(V)[np.where(Delta_hat_V[V.index(i)])[0]]
            for j in N_i:
                if ((j,i) not in Delta) and (j not in S):
                    Delta.append((i,j))
                    #print(i,j)
                    
    # Prune part
    for (i,j) in Delta:
        debug = [j in group for group in Order]
        if True not in debug:
            print(j,Order,V,Delta)
            return j,Order,V,Delta
        group_of_j = [j in group for group in Order].index(True)
        S_ij = flatten_list([Order[k] for k in range(group_of_j,len(Order))])
        S_ij.append(i)
        S_ij = sorted(S_ij)          
        Delta_hat_S, obj_hist_S = Delta_Theta_func(S1[S_ij][:,S_ij],S2[S_ij][:,S_ij],lambda_l1,rho,n_max_iter,stop_cond,verbose=False)
        if np.abs(Delta_hat_S[S_ij.index(i),S_ij.index(j)]) < th2:
            Delta.remove((i,j))
            
    Delta_B = np.zeros((p,p))
    for (i,j) in Delta:
        Delta_B[i,j] = 1
    
    I_hat = sorted(list(set(np.where(Delta_B)[1])))
    t_past = time.time() - t0
    return I_hat, Delta_B, t_past


def run_comparison_repeated(p_list,density_list,n_samples_list,I_size,n_repeat,\
                      shift=0.0,plus_variance=0.0,B_distortion_amplitude=0,perfect_intervention=False,\
                          rho=1,lambda_l1=0.2,single_threshold=0.1,pair_l1=0.1,pair_threshold=5e-3,parent_l1 = 0.1,\
                              th1=1e-3,th2=1e-3):


    I_tp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list))) 
    I_fp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    I_fn = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_tp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_fp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_fn = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    
    I_tp_ref = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    I_fp_ref = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    I_fn_ref = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_tp_ref = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_fp_ref = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_fn_ref = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    
    time_ours = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    time_ref = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))

    
    for i in range(n_repeat):
        for j in range(len(p_list)):
            for k in range(len(density_list)):
                B1,G1,mu1,variance1,Omega1,Theta1,Cov1,B2,G2,mu2,variance2,Omega2,Theta2,Cov2,Delta_Theta,S_Delta,I \
                    = create_intervention(p_list[j],I_size,density_list[k],mu=0,shift=shift,plus_variance=plus_variance,variance=1.0,\
                                      B_distortion_amplitude=B_distortion_amplitude,perfect_intervention=perfect_intervention)
                
                #diff_marginal_noise = np.abs(1/np.diag(Cov1)-1/np.diag(Cov2))
                #J0 = np.intersect1d(np.where(diff_marginal_noise<1e-6)[0],S_Delta)
                I_parents = [np.where(B1[:,i])[0].tolist() for i in I]
                #Delta_GT = Theta2-Theta1
                
                for s in range(len(n_samples_list)):
                    X1 = sample(B1,mu1,variance1,n_samples_list[s])
                    X2 = sample(B2,mu2,variance2,n_samples_list[s])
                    S1 = (X1.T@X1)/n_samples_list[s]
                    S2 = (X2.T@X2)/n_samples_list[s]
                    
                    I_hat, I_hat_parents, N_lists, A_groups, t_past = algorithm_sample(S1,S2,lambda_l1,rho,single_threshold,\
                                       pair_l1,pair_threshold,parent_l1,return_parents=True,verbose=False,Delta_hat_parent_check=True)
        
                    tp_i, fp_i, fn_i, tp_e, fp_e, fn_e = counter(I,I_hat,I_parents,I_hat_parents)
                    I_tp[i,j,k,s] = tp_i; I_fp[i,j,k,s] = fp_i; I_fn[i,j,k,s] = fn_i; 
                    e_tp[i,j,k,s] = tp_e; e_fp[i,j,k,s] = fp_e; e_fn[i,j,k,s] = fn_e
                    time_ours[i,j,k,s] = t_past


                    I_ghoshal, Delta_B_ghoshal, t2 = ghoshal_algo(S1,S2,lambda_l1,rho,th1,th2)
                    I_ghoshal_parents = [np.where(Delta_B_ghoshal[:,i])[0].tolist() for i in I_ghoshal]

                    tp_i_ref, fp_i_ref, fn_i_ref, tp_e_ref, fp_e_ref, fn_e_ref = counter(I,I_ghoshal,I_parents,I_ghoshal_parents)
                    I_tp_ref[i,j,k,s] = tp_i_ref; I_fp_ref[i,j,k,s] = fp_i_ref; I_fn_ref[i,j,k,s] = fn_i_ref
                    e_tp_ref[i,j,k,s] = tp_e_ref; e_fp_ref[i,j,k,s] = fp_e_ref; e_fn_ref[i,j,k,s] = fn_e_ref
                    time_ref[i,j,k,s] = t2
                    print(i,j,k,s)
           

    res = {'n_repeat':n_repeat,'p_list':p_list,'density_list':density_list,'I_size':I_size,\
            'n_samples_list':n_samples_list,'I_tp':I_tp, 'I_fp':I_fp, 'I_fn':I_fn, \
        'e_tp':e_tp, 'e_fp':e_fp,'e_fn':e_fn, 'time':time_ours,\
            'I_tp_ref':I_tp_ref, 'I_fp_ref':I_fp_ref, 'I_fn_ref':I_fn_ref, \
           'e_tp_ref':e_tp_ref, 'e_fp_ref':e_fp_ref,'e_fn_ref':e_fn_ref, 'time_ref':time_ref}
        
    return res

def load_res(filename,algos='ours'):
    with open(SIMULATIONS_ESTIMATED_FOLDER+'/'+filename+'.pkl','rb') as f:
        res = pickle.load(f)
        
    I_tp = res['I_tp']
    I_fp = res['I_fp']
    I_fn = res['I_fn']
    e_tp = res['e_tp']
    e_fp = res['e_fp']
    e_fn = res['e_fn']
    time_ours = res['time']
    if algos == 'ours':
        return res, I_tp, I_fp, I_fn, e_tp, e_fp, e_fn, time_ours
    elif algos == 'comparison':
        I_tp_ref = res['I_tp_ref']
        I_fp_ref = res['I_fp_ref']
        I_fn_ref = res['I_fn_ref']
        e_tp_ref = res['e_tp_ref']
        e_fp_ref = res['e_fp_ref']
        e_fn_ref = res['e_fn_ref']
        time_ref = res['time_ref']        
        return res, I_tp, I_fp, I_fn, e_tp, e_fp, e_fn, time_ours, \
            I_tp_ref, I_fp_ref, I_fn_ref, e_tp_ref, e_fp_ref, e_fn_ref, time_ref


def scores(I_tp,I_fp,I_fn,e_tp,e_fp,e_fn):
    I_precision = np.sum(I_tp,0) / (np.sum(I_tp,0)+np.sum(I_fp,0))
    I_recall =  np.sum(I_tp,0) / (np.sum(I_tp,0)+np.sum(I_fn,0))
    I_f1 = np.sum(I_tp,0) / (np.sum(I_tp,0)+(np.sum(I_fp,0)+np.sum(I_fn,0))/2)
    
    e_precision = np.sum(e_tp,0) / (np.sum(e_tp,0)+np.sum(e_fp,0))
    e_recall =  np.sum(e_tp,0) / (np.sum(e_tp,0)+np.sum(e_fn,0))
    e_f1 = np.sum(e_tp,0) / (np.sum(e_tp,0)+(np.sum(e_fp,0)+np.sum(e_fn,0))/2)
    
    return I_precision, I_recall, I_f1, e_precision, e_recall ,e_f1


#%%

rho = 1.0
lambda_l1 = 0.2    # for S_Delta estimation, and pruning
single_threshold = 0.1     # for J0 estimation
pair_l1 = 0.1               # for J0_k estimation
pair_threshold = 5e-3       # for J0_k estimation, throwaway very small ones
parent_l1 = 0.1          # for post-parent estimation     
n_max_iter = 500
stop_cond = 1e-6
verbose = False
tol = 1e-9

th1 = 5e-3
th2 = 1e-2

n_repeat = 100
p_list = [20,40,60,80,100]
density_list = [1.5,2.5]
I_size = 5
n_samples_list = [5000,10000]


res_perfect_comparison = run_comparison_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=1.0,plus_variance=0.5,\
                            B_distortion_amplitude=0.0,perfect_intervention=True,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1,th1=th1,th2=th2)
    
    
f = open(SIMULATIONS_ESTIMATED_FOLDER+'/perfect_comparison_ghoshal_2.pkl','wb')
pickle.dump(res_perfect_comparison,f)
f.close()

#%%
res_per, I_tp_per, I_fp_per, I_fn_per, e_tp_per, e_fp_per, e_fn_per, time_per, \
    I_tp_ref_per, I_fp_ref_per, I_fn_ref_per, e_tp_ref_per, e_fp_ref_per, e_fn_ref_per, time_ref_per = \
        load_res('perfect_comparison_ghoshal_2',algos='comparison')
        
        
I_precision_per, I_recall_per, I_f1_per, e_precision_per, e_recall_per, e_f1_per = \
    scores(I_tp_per, I_fp_per, I_fn_per, e_tp_per, e_fp_per, e_fn_per)
    
I_precision_ref_per, I_recall_ref_per, I_f1_ref_per, e_precision_ref_per, e_recall_ref_per, e_f1_ref_per = \
    scores(I_tp_ref_per, I_fp_ref_per, I_fn_ref_per, e_tp_ref_per, e_fp_ref_per, e_fn_ref_per)
    
time_per = np.mean(time_per,0)
time_ref_per = np.mean(time_ref_per,0)
#%%
# get the results for 5000 samples, p=20 to 100 with density 1.5. possibly add to table 1
a = 1 # density. 0 for 1.5, 1 for 2.5
b = 1 # n_samples. 0 for 5000, 1 for 10000

print('ours_precision',I_precision_per[:,a,b])
print('ours_recall',I_recall_per[:,a,b])
print('ours_f1',I_f1_per[:,a,b])
print('ours_time',time_per[:,a,b])

print('ghoshal_precision',I_precision_ref_per[:,a,b])
print('ghoshal_recall',I_recall_ref_per[:,a,b])
print('ghoshal_f1',I_f1_ref_per[:,a,b])
print('ghoshal_time',time_ref_per[:,a,b])

