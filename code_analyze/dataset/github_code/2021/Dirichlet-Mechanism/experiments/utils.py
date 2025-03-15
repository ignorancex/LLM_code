from matplotlib.ticker import SymmetricalLogLocator
from mpmath import fabs, mpf, sign
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import polygamma, loggamma, digamma
from sklearn.model_selection import train_test_split

from prior_work import compute_delta, compute_delta_binary, compute_epsilon


def logdelta(x, epsilon, r, prior, Delta_2sq, Delta_inf):
    """The objective function in RDP-to-(epsilon-delta)-DP conversion"""
    a = prior + 3*(x-1)*Delta_inf*r
    vareps_lambda = 0.5*(r**2)*x*Delta_2sq*polygamma(1, a)
    return (x-1)*(vareps_lambda - epsilon)+(x-1)*np.log(x-1)-x*np.log(x)


def Dlogdelta(x, epsilon, r, prior, Delta_2sq, Delta_inf):
    """The derivative with respect to lambda of the
    objective function in RDP-to-(epsilon-delta)-DP conversion"""
    a = prior + 3*(x-1)*Delta_inf*r
    r2d2 = (r**2)*Delta_2sq
    r2d2psi1 = r2d2*polygamma(1, a)
    r2d2psi2 = r2d2*polygamma(2, a)
    d1 = 0.5*x*r2d2psi1 - epsilon
    d2 = 0.5*r2d2psi1
    d3 = 1.5*r*x*Delta_inf*r2d2psi2
    d4 = np.log(x-1)-np.log(x)
    return d1 + (x-1)*(d2 + d3) + d4


def epsilon_func(x, epsilon, prior, lambda_, Delta_2sq, Delta_inf):
    a = prior + 3*(lambda_-1)*Delta_inf*x
    return epsilon-0.5*(x**2)*lambda_*Delta_2sq*polygamma(1, a)


def epsilon2r(epsilon, prior, lambda_=2, Delta_2sq=1, Delta_inf=1):
    """Compute r given other parameters"""
    denom = 0.5*lambda_*Delta_2sq*polygamma(1, prior)
    r0 = np.sqrt(epsilon/(denom))
    r = optimize.fsolve(epsilon_func,
                        x0=r0,
                        args=(epsilon,
                              prior,
                              lambda_,
                              Delta_2sq,
                              Delta_inf))
    return r


def r2epsilon(r, prior, lambda_=2, Delta_2sq=1, Delta_inf=1):
    """Compute epsilon given r"""
    a = prior + 3*(lambda_-1)*Delta_inf*r
    denom = 0.5*lambda_*Delta_2sq*polygamma(1, a)
    epsilon = (r**2)*denom
    return epsilon


def epsilon2adp(epsilon, lambda_, delta):
    """Convert from RDP to (epsilon-delta)-DP at a given delta"""
    adp_eps = epsilon - (np.log(delta) + lambda_*np.log(lambda_))/(lambda_-1)
    adp_eps += np.log(lambda_-1)
    return adp_eps


def adp2epsilon(eps_hat, lambda_, delta):
    """Compute epsilon given Approximate-DP parameters"""
    def func_adp(x, lambda_, delta):
        return epsilon2adp(x, lambda_, delta) - eps_hat
    epsilon = optimize.fsolve(func_adp,
                              x0=eps_hat,
                              args=(lambda_,
                                    delta)
                              )
    return epsilon[0]


def adp2epsilon_optim(eps_hat, delta):
    """Compute epsilon given epsilon_hat"""
    def func(x, delta):
        return epsilon2adp(x, 1/x+1, delta) - eps_hat
    epsilon = optimize.brentq(func,
                              a=1e-6,
                              b=eps_hat,
                              args=(delta),
                              rtol=1e-7
                              )
    return epsilon


def r2adp(r, prior, lambda_, delta, Delta_2sq=1, Delta_inf=1):
    """Compute (epsilon-delta)-DP at a given r and prior"""
    # minimize logdelta
    rdp_eps = r2epsilon(r,
                        prior,
                        lambda_,
                        Delta_2sq=1,
                        Delta_inf=1)
    adp_eps = epsilon2adp(rdp_eps, lambda_, delta)
    return adp_eps


def adp2r(epsilon, prior, lambda_, delta, Delta_2sq=1, Delta_inf=1):
    """Compute r given Approximate-DP parameters"""
    def func(r, prior, lambda_, Delta_2sq, Delta_inf):
        return r2adp(r, prior, lambda_, Delta_2sq, Delta_inf) - epsilon
    r = optimize.fsolve(func,
                        x0=1,
                        args=(prior,
                              lambda_,
                              Delta_2sq,
                              Delta_inf))
    return r[0]


def r2epsilon_evidence(x1, x2, r, alpha, lambda_):
    """Compute the Renyi divergence between Dirichlet(x1+alpha)
    and Dirichlet(x2+alpha)"""
    u1 = r*x1+alpha
    u2 = r*x2+alpha
    sumu1 = u1.sum()
    sumu2 = u2.sum()
    logE = (lambda_-1)*np.sum(loggamma(u2)-loggamma(u1)) \
        + np.sum(loggamma(u1+(lambda_-1)*(u1-u2))-loggamma(u1)) \
        - (lambda_-1)*(loggamma(sumu2)-loggamma(sumu1)) \
        - np.sum(loggamma(sumu1+(lambda_-1)*(sumu1-sumu2))-loggamma(sumu1))
    return logE/(lambda_-1)


def scale2epsilon(scale, lambda_, Delta_1):
    nts = scale / Delta_1
    log_arg = lambda_ / (2*lambda_-1)*np.exp((lambda_-1)/nts)
    log_arg += (lambda_ - 1) / (2*lambda_-1)*np.exp(-lambda_/nts)
    return 1/(lambda_-1)*np.log(log_arg)


def epsilon2scale(epsilon, lambda_, Delta_1=2):
    """Compute Laplace scale given epsilon"""
    def scale_search(scale, lambda_, Delta_1):
        return scale2epsilon(scale, lambda_, Delta_1) - epsilon
    # s0 = np.sqrt(2*lambda_/epsilon)
    s = optimize.brentq(scale_search,
                        a=lambda_/300,
                        b=lambda_*300,
                        args=(lambda_,
                              Delta_1))
    return s


def KLDir(a, b):
    """Compute the KL-divergence between Dirichlet(a) and Dirichlet(b)"""
    a0 = np.sum(a, axis=1)
    b0 = np.sum(b, axis=1)
    KL = loggamma(a0)-np.sum(loggamma(a), axis=1) \
        - loggamma(b0)+np.sum(loggamma(b), axis=1) \
        + np.sum((a-b)*(digamma(a)-digamma(a0)[:, None]), axis=1)
    return KL


def bisection(f, a, b, tol, it=0):
    '''approximates a root of f bounded
    by a and b to within tolerance tol'''

    if a > b:
        return None

    if b > 0.5:
        return None

    if sign(f(a)) == sign(f(b)):
        return bisection(f, 1.05*a, 0.95*b, tol, it)

    m = (a + b)/2

    if fabs(f(m)) < tol:
        return m
    if it > 100:
        return None
    elif sign(f(m)) == sign(f(a)):
        return bisection(f, m, b, tol, it+1)
    elif sign(f(b)) == sign(f(m)):
        return bisection(f, a, m, tol, it+1)
    else:
        return m


def r2adp_gohari(r, prior, lambda_, delta, Delta_inf, N, d):
    alpha = mpf(prior) + 4*(lambda_-1) * r * Delta_inf
    k = N * r + alpha * d
    e1 = (r + alpha) / k
    e2 = (d - 2) * alpha / k

    if d == 2:
        def func(g):
            val = compute_delta_binary(g, k, e1, e2)
            diff = val - mpf(delta)
            return diff
    else:
        def func(g):
            val = compute_delta(g, k, e1, e2, 'xiao_gimbutas_50')
            diff = val - mpf(delta)
            return diff

    gamma = bisection(func,
                      1e-7,
                      0.2,
                      tol=mpf('1e-6'))

    if gamma is not None:
        epsilon = compute_epsilon(g=gamma, k=k, W=2,
                                  e1=e1, e2=e2,
                                  ss=2*r / k)
    else:
        epsilon = None

    return epsilon


def adp2r_gohari(eps_hat, prior, lambda_, delta, Delta_inf, N, d):
    def func(r, prior, lambda_, delta, Delta_inf, N, d):
        if hasattr(r, "__len__"):
            r = r[0]
        return r2adp_gohari(r,
                            prior,
                            lambda_,
                            delta,
                            Delta_inf,
                            N,
                            d) - eps_hat
    r = optimize.fsolve(func,
                        x0=1,
                        args=(prior,
                              lambda_,
                              delta,
                              Delta_inf,
                              N,
                              d))
    print(r)
    return r[0]


class MajorSymLogLocator(SymmetricalLogLocator):

    def __init__(self):
        super().__init__(base=10., linthresh=1.)

    @staticmethod
    def orders_magnitude(vmin, vmax):

        max_size = np.log10(max(abs(vmax), 1))
        min_size = np.log10(max(abs(vmin), 1))

        if vmax > 1 and vmin > 1:
            return max_size - min_size
        elif vmax < -1 and vmin < -1:
            return min_size - max_size
        else:
            return max(min_size, max_size)

    def tick_values(self, vmin, vmax):

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        orders_magnitude = self.orders_magnitude(vmin, vmax)

        if orders_magnitude <= 1:
            spread = vmax - vmin
            exp = np.floor(np.log10(spread))
            rest = spread * 10 ** (-exp)

            stride = 10 ** exp * (0.25 if rest < 2. else
                                  0.5 if rest < 4 else
                                  1. if rest < 6 else
                                  2.)

            vmin = np.floor(vmin / stride) * stride
            return np.arange(vmin, vmax, stride)

        if orders_magnitude <= 2:
            pos_a, pos_b = np.floor(np.log10(max(vmin, 1))), np.ceil(np.log10(max(vmax, 1)))
            positive_powers = 10 ** np.linspace(pos_a, pos_b, int(pos_b - pos_a) + 1)
            positive = np.ravel(np.outer(positive_powers, [1., 5.]))

            linear = np.array([0.]) if vmin < 1 and vmax > -1 else np.array([])

            neg_a, neg_b = np.floor(np.log10(-min(vmin, -1))), np.ceil(np.log10(-min(vmax, -1)))
            negative_powers = - 10 ** np.linspace(neg_b, neg_a, int(neg_a - neg_b) + 1)[::-1]
            negative = np.ravel(np.outer(negative_powers, [1., 5.]))

            return np.concatenate([negative, linear, positive])

        else:

            pos_a, pos_b = np.floor(np.log10(max(vmin, 1))), np.ceil(np.log10(max(vmax, 1)))
            positive = 10 ** np.linspace(pos_a, pos_b, int(pos_b - pos_a) + 1)

            linear = np.array([0.]) if vmin < 1 and vmax > -1 else np.array([])

            neg_a, neg_b = np.floor(np.log10(-min(vmin, -1))), np.ceil(np.log10(-min(vmax, -1)))
            negative = - 10 ** np.linspace(neg_b, neg_a, int(neg_a - neg_b) + 1)[::-1]

            return np.concatenate([negative, linear, positive])


def symlogfmt(x, pos):
    return f'{x:.6f}'.rstrip('0')


def prepare_labeled_data(filename, test_size, num_bins, seed):
    data = pd.read_csv(filename)

    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    # remove cols with only one category
    cat_counts = X_train.nunique()
    X_train = X_train.loc[:, cat_counts != 1]
    X_test = X_test.loc[:, cat_counts != 1]
    # quantile binning while preserving repeated values
    N = X_train.shape[0]
    for col in X_train.columns:
        train_series = X_train[col].copy()
        if train_series.dtype != 'O':
            counts = train_series.value_counts()
            if counts.shape[0] > 30:
                idx = 0
                while idx < counts.shape[0] and counts.iloc[idx] > N/(3**(idx+1)):
                    idx += 1
                if idx < counts.shape[0] and counts.shape[0] - idx > 30:
                    train_remain_idx = train_series.isin(counts.index[idx:])
                    train_subset = train_series[train_remain_idx]
                    train_subset, bins = pd.qcut(train_subset,
                                                 num_bins,
                                                 retbins=True,
                                                 duplicates='drop')
                    X_train.loc[train_remain_idx, col] = train_subset

                    test_series = X_test[col].copy()
                    test_remain_idx = test_series.isin(counts.index[idx:])
                    test_subset = test_series[test_remain_idx]
                    X_test.loc[test_remain_idx, col] = pd.cut(test_subset, bins)

    X_train = X_train.astype(str)
    X_test = X_test.astype(str)
    return X_train, X_test, y_train, y_test


def prepare_data(filename, test_size, num_bins, seed):
    X = pd.read_csv(filename)

    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=seed)

    # remove cols with only one category
    cat_counts = X_train.nunique()
    X_train = X_train.loc[:, cat_counts != 1]
    X_test = X_test.loc[:, cat_counts != 1]

    # quantile binning while preserving repeated values
    N = X_train.shape[0]
    for col in X_train.columns:
        train_series = X_train[col].copy()
        if train_series.dtype != 'O':
            counts = train_series.value_counts()
            if counts.shape[0] > 30:
                idx = 0
                while idx < counts.shape[0] and counts.iloc[idx] > N/(3**(idx+1)):
                    idx += 1
                if idx < counts.shape[0] and counts.shape[0] - idx > 30:
                    train_remain_idx = train_series.isin(counts.index[idx:])
                    train_subset = train_series[train_remain_idx]
                    train_subset, bins = pd.qcut(train_subset,
                                                 num_bins,
                                                 retbins=True,
                                                 duplicates='drop')
                    X_train.loc[train_remain_idx, col] = train_subset

                    test_series = X_test[col].copy()
                    test_remain_idx = test_series.isin(counts.index[idx:])
                    test_subset = test_series[test_remain_idx]
                    X_test.loc[test_remain_idx, col] = pd.cut(test_subset, bins)

    X_train = X_train.astype(str)
    X_test = X_test.astype(str)
    return X_train, X_test


def loglikelihood(counts, params):
    ll = 0
    for node in counts.keys():
        ll += (np.log(params[node])*counts[node]).sum()

    return ll
