import numpy as np

from scipy.integrate import quad
from scipy.special import digamma

from qcd.qcd_basics import CF, CA, TF, N_F
import numpy as np

from scipy.special import digamma

from qcd.qcd_basics import CF, CA, TF, N_F


def plus_distribution(f, x, epsilon=1e-5):
    """
    Numerical implementation of the plus-distribution for a function f(x).
    Ensures the integral of [f(x)]_+ over [0, 1] is zero.

    Parameters:
        f (function): Function to regularize.
        x (float): The evaluation point.
        epsilon (float): Small regularization parameter.

    Returns:
        float: Regularized value of f(x) at x.
    """
    # Avoid numerical instability near x=0 and x=1
    x_clipped = np.clip(x, epsilon, 1 - epsilon)

    # Evaluate the integral of f(x) over [0, 1]
    integral_f, _ = quad(f, 0, 1)

    # Regularized function value
    return f(x_clipped) - (1 if x_clipped == epsilon else 0) * integral_f


def delta_function(x, epsilon):
    """Approximate delta distribution as a product of heavisides
    with total width epsilon symmetric around x=0, enforcing that
        -epsilon/2 < x < epsilon/2.

    A version which is not symmetric around x=0, with support only
    for x > 0, can easily be achieved via
        delta_function(x-epsilon/2, epsilon),
    which still has width epsilon.
    """
    return np.heaviside(x+epsilon/2, 1) \
        * np.heaviside(-(x-epsilon/2), 1) / (epsilon)


def P_qq(x, epsilon=1e-5, regularize=True):
    """LO quark to quark splitting function."""
    # Plus-regularization term -- ensures integration to 0
    reg_term = 0
    if regularize:
        # Clipping x
        x = np.clip(x, 0, 1 - epsilon)

        # Plus-regularization
        # from integral from 0 to 1-epsilon
        reg_constant = -2*np.log(epsilon) +\
                    epsilon*(2-epsilon/2) - 3/2
        # from 1-epsilon to 1, where the splitting function is fixed
        reg_constant += epsilon**2 - 2*epsilon + 2
        # adding group theory factors and delta function
        reg_term = -CF*reg_constant*delta_function(1-x-epsilon/2,
                                                   epsilon)

    # Finite term
    finite_term = CF * ((1+x**2) / (1-x))

    return finite_term + reg_term


def P_gq(x, epsilon=1e-5, regularize=True):
    """LO quark to gluon splitting function."""
    return P_qq(1-x, epsilon=epsilon, regularize=regularize)


def P_qg(x, num_flavors=N_F, **kwargs):
    """LO gluon to quark splitting function, with an extra
    factor of 2 from summing over both quarks and anti-quarks.
    """
    return TF * num_flavors * (x**2 + (1-x)**2)


def P_gg(x, epsilon=1e-5, regularize=True, num_flavors=N_F):
    """LO gluon to gluon splitting function."""
    # Plus-regularization term -- ensures integration to 0
    reg_term = 0
    if regularize:
        # Clipping x
        x = np.clip(x, epsilon, 1 - epsilon)

        # Plus-regularization
        # from epsilon to 1-epsilon
        reg_constant = (2*epsilon-1)*(11+2*epsilon*(epsilon-1))/6\
                            + 2*np.arctanh(1-2*epsilon)\
                            - np.log(epsilon) + np.log(1-epsilon)
        # from 0 to epsilon and 1-epsilon to 1
        reg_constant += 2*(epsilon**2.-epsilon+1)**2. / (1-epsilon)
        # extra 1/2 from putting contribution in 2 delta functions
        reg_constant /= 2

        # From additional delta functions (which enforce sum rule)
        # TODO: implement and DEBUG
        # reg_constant += (-11/12 + 2*TF*num_flavors/(6*CA))/2
        reg_constant += (2*TF*num_flavors/(6*CA))/2

        # Regularizing with delta-functions at both x=0 and x=1
        # (this is the reason for the earlier factors of 1/2)
        reg_term = -2*CA*reg_constant*(
                            delta_function(x-epsilon/2, epsilon) +
                            delta_function(1-x-epsilon/2, epsilon)
                         )

    # Finite term
    finite_term = 2 * CA * (x/(1-x) + (1-x)/x + x*(1-x))

    return finite_term + reg_term


def harmonic_number(N):
    return np.euler_gamma + digamma(N+1)

def jet_calc_sum(N):
    return harmonic_number(N) - 1

def Phat_qq(N):
    return CF * (3/2 - 1/N-1/(N+1) - 2*harmonic_number(N-1))

def Phat_gq(N):
    return CF * (N**2 + N + 2)/(N**3 - N)

def Phat_qg(N, num_flavors=N_F):
    return TF * N_F * (N**2 + N + 2) / (N*(N+1)*(N+2))

def Phat_gg(N, num_flavors=N_F):
    return 2*CA*(11/6*(1+N)*(2*N-1)/(N**3-N) - harmonic_number(N+2))\
        - num_flavors/6

# def Phat_gg(N, num_flavors=N_F):
#     return CA * (-1/6 + 2/((N-1)*N) + 2/((N+1)*(N+2)) \
#                  + jet_calc_sum(N)) - num_flavors/3


def Phat_matrix(N):
    return np.array([[Phat_qq(N), Phat_qg(N)],
                     [Phat_gq(N), Phat_gg(N)]])
