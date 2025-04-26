"""
generate results for shifted mean, increased variance and perfect intervention.

run for our algorithm: generate results for I recovery
run for our algorithm + UT-IGSP: generate results for I recovery (also I_parent recovery)

Use Gauss CI tests for UT-IGSP

"""

import numpy as np
import pickle
from config import SIMULATIONS_ESTIMATED_FOLDER
from functions import algorithm_sample
from functions_utigsp import run_utigsp
from helpers import create_intervention, sample, counter

#%%
def run_ours_repeated(p_list,density_list,n_samples_list,I_size,n_repeat,\
                      shift=0.0,plus_variance=0.0,B_distortion_amplitude=0,perfect_intervention=False,\
                          rho=1,lambda_l1=0.2,single_threshold=0.1,pair_l1=0.1,pair_threshold=5e-3,parent_l1 = 0.1):

    I_tp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    I_fp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    I_fn = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_tp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_fp = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    e_fn = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    
    time_ours = np.zeros((n_repeat,len(p_list),len(density_list),len(n_samples_list)))
    
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
                    print(i,j,k,s)

    res = {'n_repeat':n_repeat,'p_list':p_list,'density_list':density_list,'I_size':I_size,\
               'n_samples_list':n_samples_list,'I_tp':I_tp, 'I_fp':I_fp, 'I_fn':I_fn, \
           'e_tp':e_tp, 'e_fp':e_fp,'e_fn':e_fn, 'time':time_ours}
    
    return res

def run_comparison_repeated(p_list,density_list,n_samples_list,I_size,n_repeat,\
                      shift=0.0,plus_variance=0.0,B_distortion_amplitude=0,perfect_intervention=False,\
                          rho=1,lambda_l1=0.2,single_threshold=0.1,pair_l1=0.1,pair_threshold=5e-3,parent_l1 = 0.1,\
                              alpha=1e-3,alpha_inv=1e-3):


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

                    dag_utigsp, I_utigsp, t2 = run_utigsp(X1,X2,alpha,alpha_inv)
                    dag_utigsp = dag_utigsp.to_amat()[0]
                    I_utigsp_parents = [np.where(dag_utigsp[:,i])[0].tolist() for i in I_utigsp]
                    tp_i_ref, fp_i_ref, fn_i_ref, tp_e_ref, fp_e_ref, fn_e_ref = counter(I,I_utigsp,I_parents,I_utigsp_parents)
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
    
#%%
'''
just using our algorithm, plus_variance setting. i.e. N(0,1) to N(0,2)
p = 20,40,60,80,100,200 
'''

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

n_repeat = 50
p_list = [20,40,60,80,100,200]
density_list = [1.5,2.5]
I_size = 5
n_samples_list = [2000,3000,5000,10000]

'run ours on increased variance intervention'

res_inc = run_ours_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=0.0,plus_variance=1.0,\
                            B_distortion_amplitude=0.0,perfect_intervention=False,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1)

    
f = open(SIMULATIONS_ESTIMATED_FOLDER+'/increased_variance_1.pkl','wb')
pickle.dump(res_inc,f)
f.close()
    
#I_precision, I_recall, I_f1, e_precision, e_recall, e_f1 = scores(I_tp,I_fp,I_fn,e_tp,e_fp,e_fn)

#%%
'run ours on shifted mean intervention'

res_shift = run_ours_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=1.0,plus_variance=0.0,\
                            B_distortion_amplitude=0.0,perfect_intervention=False,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1)

    
f = open(SIMULATIONS_ESTIMATED_FOLDER+'/shifted_mean_1.pkl','wb')
pickle.dump(res_shift,f)
f.close()

        
#%%
'run ours and UT-IGSP on increased variance intervention'
n_repeat = 50
p_list = [20,40,60,80]
density_list = [1.5,2.5]
I_size = 5
n_samples_list = [5000,10000]

res_inc_comparison = run_comparison_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=1.0,plus_variance=0.0,\
                            B_distortion_amplitude=0.0,perfect_intervention=False,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1)

    
f = open(SIMULATIONS_ESTIMATED_FOLDER+'/increased_variance_comparison_1.pkl','wb')
pickle.dump(res_inc_comparison,f)
f.close()
        
#%%
'run ours and UT-IGSP on shifted mean intervention'
res_shift_comparison = run_comparison_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=1.0,plus_variance=0.0,\
                            B_distortion_amplitude=0.0,perfect_intervention=False,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1)

    

f = open(SIMULATIONS_ESTIMATED_FOLDER+'/shifted_mean_comparison_1.pkl','wb')
pickle.dump(res_shift_comparison,f)
f.close()

#%%
'run ours and UT-IGSP on shifted mean intervention'
n_repeat = 50
p_list = [100]
density_list = [1.5,2.5]
I_size = 5
n_samples_list = [5000,10000]

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

res_shift_comparison = run_comparison_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=1.0,plus_variance=0.0,\
                            B_distortion_amplitude=0.0,perfect_intervention=False,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1)

    

f = open(SIMULATIONS_ESTIMATED_FOLDER+'/shifted_mean_comparison_2.pkl','wb')
pickle.dump(res_shift_comparison,f)
f.close()


#%%
'run ours on perfect interventions.'
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

n_repeat = 50
p_list = [20,40,60,80,100,200]
density_list = [1.5,2.5]
I_size = 5
n_samples_list = [2000,3000,5000,10000]



res_perfect = run_ours_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=1.0,plus_variance=0.5,\
                            B_distortion_amplitude=0.0,perfect_intervention=True,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1)

    
f = open(SIMULATIONS_ESTIMATED_FOLDER+'/perfect_1.pkl','wb')
pickle.dump(res_perfect,f)
f.close()

#%%
'run ours and UT-IGSP on perfect interventions.'
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

n_repeat = 20
p_list = [20,40,60,80]
density_list = [1.5,2.5]
I_size = 5
n_samples_list = [5000,10000]


res_perfect_comparison = run_comparison_repeated(p_list, density_list, n_samples_list, I_size, n_repeat, shift=1.0,plus_variance=0.5,\
                            B_distortion_amplitude=0.0,perfect_intervention=True,rho=rho,lambda_l1=lambda_l1,\
                                single_threshold=single_threshold,pair_l1=pair_l1,\
                                    pair_threshold=pair_threshold,parent_l1=parent_l1)

    
f = open(SIMULATIONS_ESTIMATED_FOLDER+'/perfect_comparison_1.pkl','wb')
pickle.dump(res_perfect_comparison,f)
f.close()