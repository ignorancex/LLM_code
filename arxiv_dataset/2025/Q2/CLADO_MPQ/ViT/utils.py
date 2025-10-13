import torch,time
import numpy as np
import cvxpy as cp

format_space = [
    (
        "FP32",
        "FP32",
    ),  # calibration can still be applied but does nothing on non-integer formats
    ("INT8", "INT4"),
    ("INT8", "INT3"),
    ("INT8", "INT2"),
] 


def clado_proxy_post_process(
    proxy: dict,
    layers_to_quant: list,
    format_space: list
):
    # assign a unique index to a pair of layer and format
    layer_config_to_index = {}
    index = 0
    for layer in layers_to_quant:
        for format in format_space[1:]:
            layer_config_to_index[(layer, format)] = index
            index += 1

    # #take average over samples
    # for n in layers_to_quant:
    #     for m in layers_to_quant:
    #         for naw_index, naw in enumerate(format_space[1:]):
    #             for maw_index, maw in enumerate(format_space[1:]):
    #                 proxy[(n, m, naw_index, maw_index)] = proxy[(n, m, naw_index, maw_index)].mean()

    # convert the proxy dictionary to a 2-D proxy matrix
    proxy_matrix = np.zeros((index, index))
    for n in layers_to_quant:
        for m in layers_to_quant:
            for naw_index, naw in enumerate(format_space[1:]):
                for maw_index, maw in enumerate(format_space[1:]):
                    i = layer_config_to_index[(n, naw)]
                    j = layer_config_to_index[(m, maw)]
                    if n == m:
                        proxy_matrix[i, j] = proxy_matrix[j, i] = 2 * proxy[(n, m, naw_index, maw_index)]
                    else:
                        processed_proxy = proxy[(n, m, naw_index, maw_index)] - proxy[(n, n, naw_index, naw_index)] - proxy[(m, m, maw_index, maw_index)]
                        proxy_matrix[i, j] = proxy_matrix[j, i] = processed_proxy

    return proxy_matrix, layer_config_to_index


def qip_optimizer(
    cached_proxy,
    layer_bitops,
    layer_size,
    schemes_per_layer=3,
    bitops_bound=np.inf,
    size_bound=np.inf,
    diagonal=False,
    printInfo=False
):
    
    if cached_proxy.__class__ == torch.Tensor:
        cached_proxy = cached_proxy.cpu().numpy()

    x = cp.Variable(cached_proxy.shape[0], boolean=True)
    assert cached_proxy.shape[0]%schemes_per_layer == 0, 'cached_proxy shape[0] does not divde schemes per layer'
    num_layers = cached_proxy.shape[0] // schemes_per_layer
    
    if not diagonal:
        # convexation of cached_proxy
        es,us = np.linalg.eig(cached_proxy)
        es[es<0] = 0
        C = us@np.diag(es)@us.T
        C = (C+C.T)/2
        C = cp.atoms.affine.wraps.psd_wrap(C)
        objective = cp.Minimize(cp.quad_form(x,C))
    else:
        objective = cp.Minimize(np.diagonal(cached_proxy)@x)

    equality_constraint_matrix = []
    for i in range(num_layers):
        col = np.zeros(cached_proxy.shape[0])
        col[i*schemes_per_layer:(i+1)*schemes_per_layer] = 1
        equality_constraint_matrix.append(col)

    equality_constraint_matrix = np.array(equality_constraint_matrix)

    constraints = [equality_constraint_matrix@x == np.ones((num_layers,)),
                   layer_bitops@x/10**9<=bitops_bound,
                   layer_size@x/8/1024/1024<=size_bound]

    prob = cp.Problem(objective,constraints)
    #solver options: ['ECOS', 'ECOS_BB', 'GUROBI', 'OSQP', 'SCIP', 'SCIPY', 'SCS']
    sol_stime = time.time()
    prob.solve(solver='GUROBI', verbose=False,TimeLimit=1800) # 30minutes to solve
    sol_etime = time.time()
    sec_to_solve = sol_etime-sol_stime
    
    # Print result.
    if printInfo:
        print("Solution status: ", prob.status)
        print(f"Time to solve: {sec_to_solve:.2f}seconds")
        print("Best value found by solution: ", prob.value)
        print("The solution is:\n", x.value)
    
    return x, prob.value
