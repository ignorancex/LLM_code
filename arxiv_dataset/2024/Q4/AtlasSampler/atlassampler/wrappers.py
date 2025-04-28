import numpy as np
import os

def cmdstanpy_wrapper(draws_pd, savepath=None):

    cols = draws_pd.columns
    assert cols[9] == 'energy__'

    chain_id = draws_pd['chain__'].values.astype(int)
    tmp = draws_pd['chain__'].values
    nchains = np.unique(chain_id).size
    print(f'Number of chains in cmdstanpy data : {nchains}')

    chain_id = chain_id.reshape(nchains, -1)
    n_leapfrog = draws_pd['n_leapfrog__'].values.astype(int)
    n_leapfrog = n_leapfrog.reshape(nchains, -1)
    
    
    samples = draws_pd[cols[10:]].values
    samples = samples.reshape(nchains, -1, samples.shape[1])
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        np.save(f'{savepath}/samples', samples)
        np.save(f'{savepath}/leapfrogs', n_leapfrog)

    return samples, n_leapfrog
