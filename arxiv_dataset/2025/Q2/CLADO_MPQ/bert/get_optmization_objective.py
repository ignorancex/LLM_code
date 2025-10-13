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
        a = hm['Ltilde']
        #print(f'index {index} layer {layer_name} scheme {scheme} Ltilde {a[index,index].item():.6f}')
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
    

    #get objective for A16, W4
    x = [0,1,0] * 72

    #naive and mpqco
    #objective = np.diagonal(cached_grad) @ x

    #clado
    es,us = np.linalg.eig(cached_grad)
    es[es<0] = 0
    C = us@np.diag(es)@us.T
    C = (C+C.T)/2
    objective = x @ C @ x

    
    #print("A solution is:")
    #print(x) 
    if size_bound == 41.2158448275862:
        print("\nThe objective is", objective)


    '''
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

    prob = cp.Problem(objective,constraints)
    #solver='GUROBI'
    #['ECOS', 'ECOS_BB', 'GUROBI', 'OSQP', 'SCIP', 'SCIPY', 'SCS']
    prob.solve(solver='GUROBI', verbose=False,TimeLimit=120)
    
    # Print result.
    print("\nThe objective is", prob.value)
    print("\nSolution state", prob.status)
    print("A solution x is")
    print(x.value)
    return x
    '''

       

def feintLady(calib_num=128):

    with open(f'./clado_mpqco_results/bert-base/Ltilde_bert-base/Ltilde_calib{calib_num}_batch0(size8).pkl','rb') as f:
        hm = pickle.load(f)
    ref_layer_index = hm['layer_index']

    with open('./clado_mpqco_results/bert-base/bert-base_layer-size_layer-bitops_A16-W8-4-2.pkl', 'rb') as f:
        layer_bitops_size = pickle.load(f)
    layer_bitops = layer_bitops_size['layer_bitops']
    layer_size = layer_bitops_size['layer_size']

    #num_batches = [2, 4, 8, 16, 32, 64, 128, 256]
    num_batches = [128, ]
    for repeat in range(0,5,1):
        for n_batch in num_batches:
            if repeat >= 3 and n_batch == 256:
                continue
            else:    
                batch_Ltildes_clado = []
                # sample size: n_batch * 8
                for batch_id in range(n_batch):
                    b = batch_id + n_batch * repeat
                    with open(f'./clado_mpqco_results/bert-base/Ltilde_bert-base/Ltilde_calib{calib_num}_batch{b}(size8).pkl','rb') as f:
                        hm = pickle.load(f)
                    assert hm['layer_index'] == ref_layer_index
                    batch_Ltildes_clado.append(hm['Ltilde'])

                assert len(batch_Ltildes_clado) == n_batch
                batch_Ltildes_clado_np = np.array(batch_Ltildes_clado)
                ref_Ltilde_clado = batch_Ltildes_clado_np.mean(axis=0)

                s_batch = n_batch * repeat
                e_batch = n_batch * repeat + n_batch - 1
                n_smaples = n_batch * 8
                file_name = f'./clado_mpqco_results/bert-base/Ltilde_bert-base/multiple_batches/sample_size{n_smaples}/Ltilde_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8.pkl'
                if not os.path.exists(file_name):
                    with open(file_name, 'wb') as f:
                        pickle.dump({'Ltilde': ref_Ltilde_clado,'layer_index': ref_layer_index}, f)
                
                hm, cached_grad, index2layerscheme = load_gradients(file_name)

                batch_deltaLs_mpqco = []
                for batch_id in range(n_batch):
                    b = batch_id + n_batch * repeat
                    with open(f'./clado_mpqco_results/bert-base/DeltaL_MPQCO_bert-base/MPQCO_DELTAL_batch{b}(size8).pkl','rb') as f:
                        hm_mpqco = pickle.load(f)

                    deltal = np.zeros(ref_Ltilde_clado.shape)

                    for layer_id in range(len(index2layerscheme)):
                        layer_name,scheme = index2layerscheme[layer_id]
                        wbit = eval(scheme[:-4])[1]
                        deltal[layer_id,layer_id] = hm_mpqco[layer_name][wbit]
                    batch_deltaLs_mpqco.append(deltal)
                
                batch_deltaLs_mpqco_np = np.array(batch_deltaLs_mpqco)
                ref_deltaLs_mpqco = batch_deltaLs_mpqco_np.mean(axis=0)

                file_name = f'./clado_mpqco_results/bert-base/DeltaL_MPQCO_bert-base/multiple_batches/sample_size{n_smaples}/DeltaL_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8.pkl'
                if not os.path.exists(file_name):
                    with open(file_name, 'wb') as f:
                        pickle.dump({'DeltaL': ref_deltaLs_mpqco,'layer_index': ref_layer_index}, f)

                L = hm['Ltilde'].shape[0]

                clado_res, naive_res, mpqco_res = [], [], []
                for size_bound in np.linspace(20.2505, 81.05, 30): 
                    #print(f'Set size bound to {size_bound} MB')
                    #clado
                    '''
                    v1 = MIQCP_optimize(cached_grad=cached_grad,
                                layer_bitops=layer_bitops,
                                layer_size=layer_size,
                                schemes_per_layer=3,
                                bitops_bound=np.inf,size_bound=size_bound,
                                naive=False)
                    clado_res.append(v1)'''
                    
                    #naive
                    v2 = MIQCP_optimize(cached_grad=cached_grad,
                                layer_bitops=layer_bitops,
                                layer_size=layer_size,
                                schemes_per_layer=3,
                                bitops_bound=np.inf,size_bound=size_bound,
                                naive=True)
                    naive_res.append(v2)

                    '''
                    #mpqco
                    v3 = MIQCP_optimize(cached_grad=ref_deltaLs_mpqco,
                                layer_bitops=layer_bitops,
                                layer_size=layer_size,
                                schemes_per_layer=3,
                                bitops_bound=np.inf,size_bound=size_bound,
                                naive=True)
                    mpqco_res.append(v3)
                    '''

                with open(f'./clado_mpqco_results/bert-base/Clado_Naive_MPQCO_optimization_bert-base/sample_size{n_smaples}/clado_naive_mpqco_a16_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8_optimization.pkl','wb') as f:
                    pickle.dump({'clado_res': clado_res, 'naive_res': naive_res, 'mpqco_res': mpqco_res}, f)
            
def main():
    feintLady()


if __name__ == "__main__":
    main()

