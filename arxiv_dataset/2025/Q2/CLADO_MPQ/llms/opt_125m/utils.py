import torch,time
import numpy as np
import cvxpy as cp

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
    prob.solve(solver='GUROBI', verbose=False)
    
    # Print result.
    if printInfo:
        print("The optimal value is: ", prob.value)
        print("The solution is:\n", x.value)
    
    return x, prob.value
