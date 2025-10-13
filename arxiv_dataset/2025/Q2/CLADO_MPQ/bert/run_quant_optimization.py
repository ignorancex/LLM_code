#!/usr/bin/env python

import torch
import cvxpy as cp
import pickle
import numpy as np
import os


def load_gradients(file_name):
    with open(file_name,'rb') as f:
        hm = pickle.load(f)

    L = hm['Ltilde'].shape[0]
    index2layerscheme = [None for i in range(L)]

    for name in hm['layer_index']:
        index = hm['layer_index'][name]
        layer_name = name[:-11]
        scheme = name[-11:]
        index2layerscheme[index] = (layer_name,scheme)
    
    cached_grad = np.zeros_like(hm['Ltilde'])
    for i in range(L):
        for j in range(L):
            layer_i,scheme_i = index2layerscheme[i]
            layer_j,scheme_j = index2layerscheme[j]
            if layer_i == layer_j:
                if scheme_i == scheme_j:
                    cached_grad[i,j] = cached_grad[j,i] = 2 * hm['Ltilde'][i,j]
                else:
                    cached_grad[i,j] = cached_grad[j,i] = 0
            else:
                cached_grad[i,j] = cached_grad[j,i] = hm['Ltilde'][i,j] - hm['Ltilde'][i,i] - hm['Ltilde'][j,j]

    return hm, cached_grad, index2layerscheme


def MIQCP_optimize(cached_grad,layer_bitops,layer_size,
                   schemes_per_layer=3,
                   bitops_bound=np.inf,size_bound=np.inf,
                   naive=False):
    
    if cached_grad.__class__ == torch.Tensor:
        cached_grad = cached_grad.cpu().numpy()

    x = cp.Variable(cached_grad.shape[0], boolean=True)
    assert cached_grad.shape[0]%schemes_per_layer == 0, 'cached_gradient shape[0] does not divde schemes per layer'
    num_layers = cached_grad.shape[0]//schemes_per_layer

    if not naive:
        # convexation of cached_grad
        es,us = np.linalg.eig(cached_grad)
        es[es<0] = 0
        C = us@np.diag(es)@us.T
        C = (C+C.T)/2
        C = cp.atoms.affine.wraps.psd_wrap(C)
        objective = cp.Minimize(cp.quad_form(x,C))

    else:
        objective = cp.Minimize(np.diagonal(cached_grad)@x)

    equality_constraint_matrix = []
    for i in range(num_layers):
        col = np.zeros(cached_grad.shape[0])
        col[i*schemes_per_layer:(i+1)*schemes_per_layer] = 1
        equality_constraint_matrix.append(col)

    equality_constraint_matrix = np.array(equality_constraint_matrix)

    constraints = [equality_constraint_matrix@x == np.ones((num_layers,)),
                   layer_bitops@x/10**9<=bitops_bound,
                   layer_size@x/8/1024/1024<=size_bound]

    # x = [0,1,0]*72
    # objective = np.diagonal(cached_grad) @ x
    # return x, objective
    
    prob = cp.Problem(objective,constraints)
    #solver='GUROBI'
    #['ECOS', 'ECOS_BB', 'GUROBI', 'OSQP', 'SCIP', 'SCIPY', 'SCS']
    prob.solve(solver='GUROBI', verbose=False, TimeLimit=120)
    
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("\nSolution state", prob.status)
    print("A solution x is")
    print(x.value)
    return x, prob.value

       

def feintLady(calib_num=128):
    file_name = f'./clado_mpqco_results/bert-base-uncased-squad-v1/CLADO/Ltilde_bert-base-uncased-squad-v1/sample_size16/Ltilde_calib128_batches_0-1_bs8.pkl'
    with open(file_name,'rb') as f:
        hm = pickle.load(f)
    ref_layer_index = hm['layer_index']
    _, _, index2layerscheme = load_gradients(file_name)

    with open('bert-base-uncased-squad-v1_layer-size_layer-bitops_A16-W8-4-2.pkl', 'rb') as f:
        layer_bitops_size = pickle.load(f)
    layer_bitops = layer_bitops_size['layer_bitops']
    layer_size = layer_bitops_size['layer_size']


    num_batches = [768, ]
    for repeat in range(0,1,1):
        for n_batch in num_batches:
            if (repeat >= 3 and n_batch == 256):
                continue
            else:
                s_batch = n_batch * repeat
                e_batch = n_batch * repeat + n_batch - 1
                n_smaples = n_batch * 8

                # file_name = f'./clado_mpqco_results/bert-base-uncased-squad-v1/MPQCO/DeltaL_MPQCO_bert-base-uncased-squad-v1/sample_size{n_smaples}/DeltaL_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8.pkl'
                # with open(file_name,'rb') as f:
                #     hm_mpqco = pickle.load(f)
                # ref_deltaLs_mpqco = hm_mpqco['DeltaL']

                # file_name = f'./clado_mpqco_results/bert-base-uncased-squad-v1/HAWQ/DeltaL_HAWQ_bert-base-uncased-squad-v1/sample_size{n_smaples}/DeltaL_HAWQ_batches_{s_batch}-{e_batch}_bs8.pkl'
                # with open(file_name, 'rb') as f:
                #     hm_hawq = pickle.load(f)
                # ref_deltaLs_hawq = hm_hawq['DeltaL']

                L = hm['Ltilde'].shape[0]

                file_name = f'./clado_mpqco_results/bert-base-uncased-squad-v1/CLADO/Ltilde_bert-base-uncased-squad-v1/sample_size{n_smaples}/Ltilde_calib128_batches_{s_batch}-{e_batch}_bs8.pkl'
                _, cached_grad, _ = load_gradients(file_name)
                
                clado_res, naive_res, mpqco_res, hawq_res = [], [], [], []
                clado_objective, naive_objective, mpqco_objective, hawq_objective = [], [], [], []
                #size_bounds = np.linspace(35, 45, 15)
                size_bounds = [20.2505, 22.50055556, 24.75061111, 27.00066667, 29.25072222, 31.50077778, 33.75083333, 36.00088889, 38.25094444, 40.501, 43.39735714, 46.29371429, 49.19007143, 52.08642857, 54.98278571, 57.87914286, 60.7755, 63.67185714, 66.56821429, 69.46457143, 72.36092857, 75.25728571, 78.15364286, 81.05]
                for size_bound in size_bounds:
                    print(f'Set size bound to {size_bound} MB')
                    # clado
                    v1, objective_val = MIQCP_optimize(cached_grad=cached_grad,
                                                        layer_bitops=layer_bitops,
                                                        layer_size=layer_size,
                                                        schemes_per_layer=3,
                                                        bitops_bound=np.inf,size_bound=size_bound,
                                                        naive=False)
                    clado_res.append(v1)
                    clado_objective.append(objective_val)
                    
                    # # naive
                    # v2, objective_val = MIQCP_optimize(cached_grad=cached_grad,
                    #                                 layer_bitops=layer_bitops,
                    #                                 layer_size=layer_size,
                    #                                 schemes_per_layer=3,
                    #                                 bitops_bound=np.inf,size_bound=size_bound,
                    #                                 naive=True)
                    # naive_res.append(v2)
                    # naive_objective.append(objective_val)

                    # # mpqco
                    # v3, objective_val = MIQCP_optimize(cached_grad=ref_deltaLs_mpqco,
                    #                     layer_bitops=layer_bitops,
                    #                     layer_size=layer_size,
                    #                     schemes_per_layer=3,
                    #                     bitops_bound=np.inf,size_bound=size_bound,
                    #                     naive=True)
                    # mpqco_res.append(v3)
                    # mpqco_objective.append(objective_val)

                    # # hawq
                    # v4, objective_val = MIQCP_optimize(cached_grad=ref_deltaLs_hawq,
                    #                                 layer_bitops=layer_bitops,
                    #                                 layer_size=layer_size,
                    #                                 schemes_per_layer=3,
                    #                                 bitops_bound=np.inf,size_bound=size_bound,
                    #                                 naive=True)
                    # hawq_res.append(v4)
                    # hawq_objective.append(objective_val)



                with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/CLADO/Clado_optimal_decisions_bert-base-uncased-squad-v1/sample_size{n_smaples}/clado_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization.pkl','wb') as f:
                    pickle.dump({'clado_res': clado_res}, f)
                with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/CLADO/Clado_optimal_objectives_bert-base-uncased-squad-v1/sample_size{n_smaples}/clado_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization_objective.pkl','wb') as f:
                    pickle.dump({'clado_objectives': clado_objective}, f)    
                # with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/MPQCO/MPQCO_optimal_decisions_bert-base-uncased-squad-v1/sample_size{n_smaples}/mpqco_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization.pkl','wb') as f:
                #     pickle.dump({'mpqco_res': mpqco_res}, f)
                # with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/MPQCO/MPQCO_optimal_objectives_bert-base-uncased-squad-v1/sample_size{n_smaples}/mpqco_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization_objective.pkl','wb') as f:
                #     pickle.dump({'mpqco_objectives': mpqco_objective}, f) 
                # with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/Naive/Naive_optimal_decisions_bert-base-uncased-squad-v1/sample_size{n_smaples}/naive_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization.pkl','wb') as f:
                #     pickle.dump({'naive_res': naive_res}, f)
                # with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/Naive/Naive_optimal_objectives_bert-base-uncased-squad-v1/sample_size{n_smaples}/naive_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization_objective.pkl','wb') as f:
                #     pickle.dump({'naive_objectives': naive_objective}, f) 
                # with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/HAWQ/Hawq_optimal_decisions_bert-base-uncased-squad-v1/sample_size{n_smaples}/hawq_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization.pkl','wb') as f:
                #     pickle.dump({'hawq_res': hawq_res}, f)
                # with open(f'./clado_mpqco_results/bert-base-uncased-squad-v1/HAWQ/Hawq_optimal_objectives_bert-base-uncased-squad-v1/sample_size{n_smaples}/hawq_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization_objective.pkl','wb') as f:
                #     pickle.dump({'hawq_objectives': hawq_objective}, f) 
def main():
    feintLady()


if __name__ == "__main__":
    main()
