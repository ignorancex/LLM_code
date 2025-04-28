import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree as KDTree


def mad(data):
    return np.mean(np.abs(data - np.mean(data, axis=0)), axis=0)


def regression_ABC(s_obs, param, sumStats, p, gamma_vector):
    if param.shape[0] < param.shape[1]:
        param = np.transpose(param)

    if sumStats.shape[0] < sumStats.shape[1]:
        sumStats = np.transpose(sumStats)
    
    M = len(param)
    M_epsilon = int(M*p)

    if np.sum(gamma_vector) == 0:
        return param[:M_epsilon]
    else:
        sumStats = sumStats[:,gamma_vector]
        s_obs = s_obs[:,gamma_vector]

        norm_factor = mad(sumStats)

        norm_sumStats = sumStats / norm_factor
        norm_s_obs = s_obs / norm_factor

        distance = np.linalg.norm(norm_sumStats - norm_s_obs, axis = 1)
        max_accepted_distance = np.sort(distance)[M_epsilon - 1]

        posterior_samples = param[distance <= max_accepted_distance, :]
        norm_sumStats_star = norm_sumStats[distance <= max_accepted_distance, :]

        weights = 1 - (distance[distance <= max_accepted_distance] / max_accepted_distance)**2
        W = np.diag(weights)

        s_obs_norm = np.tile(norm_s_obs, (M_epsilon,1))
        X = np.column_stack((np.ones(shape = (M_epsilon,1)), norm_sumStats_star - s_obs_norm))

        A = np.matmul(X.T, W)

        solution = np.linalg.solve(np.matmul(A, X), np.matmul(A, posterior_samples))

        beta = solution[1:,:]

        posterior_samples_adjusted = posterior_samples - np.matmul((norm_sumStats_star - s_obs_norm), beta)

        return posterior_samples_adjusted
    
    
# Function to sample from ABC posterior given feedback
def sample_ABC_Posterior(s_obs, param, sumStats, p, gamma_vector):
    nParam = param.shape[1]
    nSamples = gamma_vector.shape[0]
    
    target_samples = np.zeros((nSamples, nParam))
    
    g_unique, g_counts = np.unique(gamma_vector, axis = 0, return_counts = True)
    c = 0

    for i in range(0,len(g_unique)):
        samples_abc = regression_ABC(s_obs, param, sumStats, p, g_unique[i,:])
        target_samples[c:c+g_counts[i],:] = gaussian_kde(samples_abc.T).resample(g_counts[i]).T
        c = c+g_counts[i]

    return target_samples


def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.

  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.

  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).

  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def sample_gamma(queried_idx, feedbacks, nSamples, nStats, pi, rho):
    delta = pi*rho + (1-pi)*(1-rho)
    gamma = np.zeros((nSamples, nStats))
    idx = list(set(list(range(0,nStats))) - set(queried_idx))
    gamma[:,idx] = np.random.rand(nSamples,len(idx))<rho
    for i in range(0,len(queried_idx)):
        f = feedbacks[i]
        z = ((pi**f)*((1-pi)**(1-f))*rho)/((delta**f)*((1-delta)**(1-f)))
        gamma[:,queried_idx[i]] = np.random.rand(nSamples)<z
    
    return gamma.astype('bool')


def generate_feedback(gamma_true, pi, index):
    g = gamma_true[index]
    return np.random.rand()<((pi**g)*((1-pi)**(1-g)))