import networkx
import numpy as np
import ot
import scipy as sp
from networkx.generators.community import stochastic_block_model as sbm


def generate_sample(
        n1,
        n2,
        seed=0,
        mu_s=np.array([0, 0]),
        cov_s=np.array([[1, 0], [0, 1]]),
        mu_t=np.array([4, 4, 4]),
        cov_t=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    """Based on
    https://pythonot.github.io/auto_examples/gromov/plot_gromov.html. Sample two
    Gaussian distributions (in 2D and 3D) and its Euclidean cost matrices, then
    convert to float16 to save memory.
    """

    xs = ot.datasets.make_2D_samples_gauss(
        n1, mu_s, cov_s, random_state=seed
    ).astype(np.float16)
    P = sp.linalg.sqrtm(cov_t)
    xt = np.random.randn(n2, 3).dot(P) + mu_t

    # Distance matrices
    C1 = sp.spatial.distance.cdist(xs, xs).astype(np.float32)
    C2 = sp.spatial.distance.cdist(xt, xt).astype(np.float32)

    # Generate uniform measures
    p = ot.unif(n1)
    q = ot.unif(n2)

    return C1, C2, p, q, xs, xt


def generate_sample_two(m, n, seed=0,
                        p2=[[1., 0.1], [0.1, 0.9]],
                        p3=[[1., 0.1, 0.], [0.1, 0.95, 0.1], [0., 0.1, 0.9]]):
    # 2 blocks SBM (Stochastic Block Model) to 3 blocks SBM
    # From https://pythonot.github.io/auto_examples/gromov/plot_fgw_solvers.html

    G2 = sbm(seed=seed, sizes=[m // 2, m // 2], p=p2)
    G3 = sbm(seed=seed, sizes=[n // 3, n // 3, n // 3], p=p3)
    part_G2 = [G2.nodes[i]['block'] for i in range(m)]
    part_G3 = [G3.nodes[i]['block'] for i in range(n)]

    C2 = networkx.to_numpy_array(G2)
    C3 = networkx.to_numpy_array(G3)

    # We add node features with given mean - by clusters
    # and inversely proportional to clusters' intra-connectivity

    F2 = np.zeros((m, 1)).astype(np.float16)
    for i, c in enumerate(part_G2):
        F2[i, 0] = np.random.normal(loc=c, scale=0.01)

    F3 = np.zeros((n, 1)).astype(np.float16)
    for i, c in enumerate(part_G3):
        F3[i, 0] = np.random.normal(loc=2. - c, scale=0.01)

    h2 = (np.ones(C2.shape[0]) / C2.shape[0])
    h3 = (np.ones(C3.shape[0]) / C3.shape[0])

    return C2, C3, h2, h3, F2, F3, G2, G3
