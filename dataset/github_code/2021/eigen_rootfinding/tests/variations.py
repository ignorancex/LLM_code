import eigen_rootfinding as er
import numpy as np
from scipy import linalg as la
import time

def report_residuals(polys,roots,kinds=['neglog10','max']):
    """
    'raw': raw residuals
    'log10': log10 residuals
    'max': max residual
    """
    evals = np.abs([[p(r) for p in polys] for r in roots])
    abspolys = [er.MultiPower(np.abs(p.coeff)) for p in polys]
    absevals = np.array([[p(r) for p in abspolys] for r in np.abs(roots)])
    r = evals/(absevals+1)
    r = np.mean(r,axis=1)
    wherezero = np.where(r == 0)[0]
    r[wherezero] = np.finfo(np.float64).eps #set to macheps #todo think about...
    infodict = {'raw':r,
                'log10':np.log10(r),
                'max':np.max(r)}
    return {kind:infodict[kind] for kind in kinds}

def run_tests(deg,dim,methods=['qrpmac',
                               'svdmac',
                               'lqmac',
                               'qrpnull',
                               'svdnull',
                               'lqnull',
                               'qrpmac_randcombs',
                               'svdmac_randcombs',
                               'lqmac_randcombs',
                               'qrpnull_randcombs',
                               'svdnull_randcombs',
                               'lqnull_randcombs',
                               'qrpfastnull',
                               'svdfastnull',
                               'lqfastnull'],
               tests=range(600,650),residualkinds=['log10','raw','max'],verbose=False):
    data = {method:{kind:[] for kind in residualkinds} for method in methods}
    for method,method_data in zip(methods,data):
        randcombos = method[-5:] == 'combs'
        if randcombos: runmethod = method[:-10]
        else: runmethod = method
        data[method]['time'] = []
        for seed in tests:
            if verbose: print('\rdim',dim,'deg',deg,method,'seed',seed,end='')
            np.random.seed(seed)
            starttime = time.time()
            polys = [er.polynomial.getPoly(deg,dim,power=True) for polynum in range(dim)]
            t = time.time()-starttime
            roots = er.solve(polys,method=runmethod,randcombos=randcombos,max_cond_num=np.inf)
            residual_info = report_residuals(polys,roots,kinds=residualkinds)
            data[method]['time'].append(t)
            for kind in residualkinds:
                data[method][kind].append(residual_info[kind])
    return data
