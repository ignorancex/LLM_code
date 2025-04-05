# The goal of this module is to implement (and test) the construction
# outlined in the proof of Lemma 15 in the paper.
#
# Given M symmetric positive definite in R^{k X k} such that
#
# tr M = m > k, (m an integer),
#
# it finds A such that:
#
# A's columns have unit norm, and
# AA^t = M.
#
# The required functionality is found in method get_A below.

# The method generic generates many PSD matrices, constructs their
# corresponding A, verifies whether they display clusterization and
# saves results to the file simulations.csv.

import numpy as np
import pandas as pd
from pandas import isnull
from numpy import sin, cos
from scipy.stats import ortho_group
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns

EPS = 1e-5 ## Numerical tolerance
ortho = ortho_group.rvs ## Generates random orthogonal matrices
cot = lambda x: cos(x) / sin(x)


def conj(D, U):
    ''' Returns UDU^t'''
    assert U.shape == D.shape
    return np.einsum('ij, jk, lk -> il', U, D, U)


def givens(theta, dims, lower, upper):
    """Returns a Givens rotation matrix

    """
    
    assert theta >= 0 and theta <= np.pi
    assert lower < upper
    assert upper <= dims
    
    R = np.eye(dims)
    R[lower, lower] = cos(theta)
    R[upper, upper] = cos(theta)
    R[lower, upper] = -sin(theta)
    R[upper, lower] = sin(theta)

    return R

   
def MDUS(m, k):
    """Generate and return random:
    M symmetric positive definite
    D diagonal with decreasing diagonal entries
    U orthogonal such that M = UDU^t
    S of shape (k,m) with diagonal that is the square root of D
    """

    
    assert m >= k

    D = np.random.lognormal(mean=50, sigma=15, size=k)
    D = np.sort(D)[::-1]
    # D = np.array([np.random.lognormal(mean=i, sigma=1) for i in range(k)]) 
    D = D * m / np.sum(D)
    assert abs(np.sum(D) - m) < EPS
    assert np.all(D > 0)
    D = np.diag(D)
    assert D.shape == (k, k)
    assert np.all(~isnull(D))

    U = ortho(k)
    assert np.all(np.abs(np.dot(U, U.T) - np.identity(k)) < EPS)

    M = conj(D, U)
    M = (M + M.T) / 2  ## Ensure it is symmetric

    S = np.zeros((k, m))
    for i in range(k):
        S[i, i] = np.sqrt(D[i, i])

    assert M.shape == D.shape == U.shape == (k, k)
    assert S.shape == (k, m)

    return M, D, U, S


def get_theta(C, upper):
   
    if upper >= C.shape[0]:
        raise ValueError(f"1: {upper}\n\n\n{C}")
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"2: {C.shape}")
    if upper == 0:
        raise ValueError(f'3: Why upper == 0? C.shape == {C.shape}.')
    if abs(C[upper, upper]) < EPS:
        return None, None

    success = False
    for lower in reversed(range(upper)):
        ckk = C[upper, upper]
        cpp = C[lower, lower]
        if ckk * cpp < 0:
            success = True
            break
    # assert success, f'4: ckk = {ckk:2.3f}, cpp = {cpp:2.3f}, sum = {ckk + cpp:2.3f}'

    ## Comment refers to common names in quadratic equation lingo
    c2 = C[upper, upper]
    c1 = 2 * C[upper, lower]
    c0 = C[lower, lower]
    # assert c2 * c0 < 0, f"5: {c2}    {c0}"

    ## Solving the quadratic to get cot(theta), which will soon
    ## be inverted to find theta.
    disc = c1 ** 2 - 4 * c2 * c0  # Discriminant
    # assert disc >= 0, f"6: {disc}"
    cot1 = (-c1 + np.sqrt(disc)) / 2 / c2
    cot2 = (-c1 - np.sqrt(disc)) / 2 / c2  ## Second solution

    theta1 = np.arctan(1 / cot1)
    theta2 = np.arctan(1 / cot2)
    if 0 <= theta1 <= np.pi:
        return theta1, lower
    elif 0 <= theta2 <= np.pi:
        return theta2, lower
    else:
        raise ValueError(f'7: Thetas == {theta1:2.2f}, {theta2:2.2f} not in [0,pi]')
        

def caller(m, k):

    print(f'm={m}, k={k}')
    assert m >= k
    M, _, _, _ = MDUS(m, k)

    A = get_A(M, m)
    assert A.shape == (k, m)

    ## Check that AAt = M, as stated.
    AAt = A @ A.T
    assert np.all(np.abs(AAt - M) < EPS)

    ## Check that A does have unit norm columns, as stated.
    norms = np.linalg.norm(A, axis=0)
    assert len(norms) == m
    assert np.all(np.abs(norms - 1) < EPS)

    # for vec in A.T:
    #     pos_ind = np.all(np.abs(A.T - vec) < EPS, axis=1)
    #     neg_ind = np.all(np.abs(A.T + vec) < EPS, axis=1)
    #     ind = np.logical_or(pos_ind, neg_ind).astype(int)
    #     assert ind.shape == (A.shape[1],)
    #     if np.sum(ind) > 1:
    #         print(ind)


def get_A(M,
          m):
    """Implement the construction in Lemma ...

    Parameters
    M: PSD matrix 
    m: integer

    Returns 
    
    A: matrix of shape (M.shape[0], m) such that A has unit norm
    columns, i.e. np.linalg.norm(A, axis=0) is all ones and A @ A.T ==
    M to numerical precision

    """
    k = M.shape[0]
    assert abs(m - np.trace(M)) < EPS

    D, U = np.linalg.eigh(M)
    D = np.maximum(D, 0)
    assert np.all(D >= -EPS), D.min()

    S = np.zeros((k, m))
    for i in range(k):
        S[i, i] = np.sqrt(D[i])

    C = np.dot(S.T, S) - np.identity(m)
    V = np.identity(m)
    for upper in reversed(range(1, m)):
        assert abs(np.trace(C)) < EPS, np.trace(C)
        theta, lower = get_theta(C, upper)
        if theta is None and lower is None:
            continue
        R = givens(theta=theta, dims=m, lower=lower, upper=upper)
        C = conj(C, R)
        V = np.dot(R, V)

    A = np.einsum('ij, jk, lk ->il', U, S, V)

    ## Verify C has zero trace
    assert abs(np.trace(C)) < EPS

    ## Verify V is orthogonal
    assert np.all(np.abs(np.diag(np.dot(V, V.T)) - 1) < EPS)
    assert np.all(np.abs(np.dot(V, V.T) - np.identity(m)) < EPS)

    return A


def simulate(m, k):
    """   
    Generate a random D-optimal design, following Lemma from paper.
    
    Parameters

    m: number of measurement points
    k: Rank of O^*O

    Returns
    True if the design is clustered, False o.w.

    """
    
    ## Random matrices such that:
    ## M is symmetric positive definite
    ## m = number of measurements
    ## Number of nonzero entries in D
    M, _, _, _ = MDUS(m, k)
                
    ## AA^t = M, and A has unit norm columns.
    A = get_A(M, m)
    
    ## distances between columns of A
    distances = cdist(A.T, A.T)
    
    ## Clusterization does not occur if all off-diagonal
    ## distances are large. Diagonal has 0 entries, but
    ## obviously that does not mean clusterization
    ## occurs. Since we do not want to count the diagonal
    ## entries as clustered, we fill diagonal with 1's
    np.fill_diagonal(distances, 1)

    ## Which distances are > 0 <==> which pairs of
    ## measurements are ***not*** clustered
    dis = distances > EPS
    
    ## If all distances are large, then measurements are not clustered
    ## and the design does not exhibit sensor clusterization.
    return {'m': m, 'k': k, 'cluster': not dis.all()}
    

def generic():
    """Run randomized simulations of D-optimal design with our model.

    """

    ## Number of simulations per pair m, k.
    N = 5000
   
    def generator(N):
        for m in range(4, 25):
            for k in range(2, m):
                for i in range(N):
                    yield m, k
                    
    pairs = tqdm(list(generator(N)))
    res = Parallel(n_jobs=-3)(delayed(simulate)(*pair) for pair in pairs)
    res = pd.DataFrame(res)
    res = res.groupby(['m', 'k']).cluster.mean()
    res.to_csv("simulations.csv")
  

def main():
    dd = pd.read_csv('simulations.csv')
    dd['mmk'] = dd.m - dd.k
    

    b = sns.lineplot(data=dd,
                     x='mmk',
                     y='cluster',
                     hue='m')


    
    b.set_xlabel("m - k",fontsize=22)
    b.set_ylabel("Clusterization Fraction",fontsize=22)
    b.tick_params(labelsize=12)
    plt.setp(b.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(b.get_legend().get_title(), fontsize='23') # for legend title

    plt.tight_layout()
    
    plt.savefig("latex/simulations.png")
    plt.show()


if __name__ == '__main__':
    try:
        generic() 
        main()
    except:
        import pdb, traceback, sys
        traceback.print_exc()
        _, _ , tb = sys.exc_info()        
        pdb.post_mortem(tb)



