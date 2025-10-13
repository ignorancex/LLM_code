import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import root
from polynomials import f as F, g as G, Dfx, Dfz, Dgx, Dgz, yfun, tfun, ufun, \
    vfun, wfun, kfun, e3fun


# ----------------------------------------------------------- General utilities
def valid_distribution(e1, e2, e3):
    """
    Checks if the given correlators `e1`, `e2`, and `e3` satisfy the positivity
    constraints required for a valid probability distribution.

    The function evaluates a set of inequalities derived from the requirement that
    the probabilities of all measurement outcomes must be non-negative. These
    inequalities define the valid region for the correlators in the parameter space.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The three-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    bool or numpy.ndarray
        - If the inputs are floats, returns `True` if the correlators satisfy the
          positivity constraints, and `False` otherwise.
        - If the inputs are numpy arrays, returns a boolean array of the same shape,
          where each element indicates whether the corresponding set of correlators
          satisfies the constraints.

    Notes:
    ------
    - The constraints are derived from the requirement that the probabilities of all
      measurement outcomes must be non-negative.
    - If `e1`, `e2`, and `e3` are numpy arrays, they must have the same shape.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2, e3 = 0.5, 0.5, 0.25
    >>> is_valid = valid_distribution(e1, e2, e3)
    >>> print("Is the distribution valid?", is_valid)

    >>> # Example with numpy arrays
    >>> e1 = np.array([0.5, 0.6])
    >>> e2 = np.array([0.5, 0.4])
    >>> e3 = np.array([0.25, 0.3])
    >>> is_valid = valid_distribution(e1, e2, e3)
    >>> print("Are the distributions valid?", is_valid)
    """
    valid = e2 >= 2*e1 - 1
    valid &= e2 >= -2*e1 - 1
    valid &= e3 >= -1 + e1 + e2
    valid &= e3 <= 1 + e1 - e2
    valid &= e3 >= -1 - 3*e1 - 3*e2
    valid &= e3 <= 1 - 3*e1 + 3*e2
    return valid


def bit_flip(e1, e2, e3):
    """
    Applies a bit-flip transformation to the correlators `e1`, `e2`, and `e3`.

    The bit-flip transformation negates `e1` and `e3` while leaving `e2` unchanged.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The three-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    tuple
        A tuple containing the transformed correlators `(-e1, e2, -e3)`. If the inputs
        are numpy arrays, the transformation is applied element-wise.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2, e3 = 0.5, 0.5, 0.25
    >>> transformed = bit_flip(e1, e2, e3)
    >>> print("Transformed correlators:", transformed)

    >>> # Example with numpy arrays
    >>> e1 = np.array([0.5, 0.6])
    >>> e2 = np.array([0.5, 0.4])
    >>> e3 = np.array([0.25, 0.3])
    >>> transformed = bit_flip(e1, e2, e3)
    >>> print("Transformed correlators:", transformed)
    """
    return -e1, e2, -e3


def fun(x, point):
    """
    Defines a system of equations to solve for the parameters `x` and `z` of the cyan surface,
    given the correlators `e1` and `e2`.

    This function evaluates two equations, `F(x, z, e1, e2)` and `G(x, z, e1, e2)`, which
    define the equations on the parameters `x` and `z` for points lying on the cyan surface.
    The system is designed to be solved using numerical solvers like `scipy.optimize.root`.

    Parameters:
    -----------
    x : tuple or numpy.ndarray
        A tuple or array containing the parameters `x` and `z` of the cyan surface.
        These are the variables to be solved for.
    point : tuple or numpy.ndarray
        A tuple or array containing the correlators `e1` and `e2`. These are fixed
        values used in the evaluation of the equations.

    Returns:
    --------
    tuple
        A tuple containing the evaluated values of the equations `F(x, z, e1, e2)`
        and `G(x, z, e1, e2)`. These values represent the residuals of the equations,
        which should be zero if evaluated in a solution.

    Notes:
    ------
    - The functions `f` and `g` are defined in the polynomials.py file. They
      represent the equations that define the cyan surface in terms of the
      parameters `x` and `z` and the correlators `e1` and `e2`.
    - This function is intended to be used with numerical solvers like `scipy.optimize.root`
      to find values of `x` and `z` that satisfy both equations simultaneously.
    - In general, this system has multiple solutions, so one has to check if the
      positivity constraints are satisfied.

    Examples:
    ---------
    >>> from scipy.optimize import root
    >>> from triangle_inequalities import fun, e3fun
    >>> e1, e2 = 1/3, -5/27
    >>> x, z = root(fun, (0.25, 0.25), args=((e1, e2), )).x
    >>> print(f'x = {x}, z = {z}')
    >>> print(f'E3 = {e3fun(x, z, e1, e2)}')
    """
    return F(*x, *point), G(*x, *point)


def jac(x, point):
    """
    Computes the Jacobian matrix of the system of equations defined in `fun`.

    The Jacobian matrix is a 2x2 matrix containing the partial derivatives of the
    equations `F(a, c, e1, e2)` and `G(a, c, e1, e2)` with respect to the variables
    `a` and `c`. This matrix is used to expedite numerical solvers like
    `scipy.optimize.root` by providing information about the local behavior of the
    system of equations.

    Parameters:
    -----------
    x : tuple or numpy.ndarray
        A tuple or array containing the parameters `a` and `c` of the cyan surface.
        These are the variables with respect to which the partial derivatives are
        computed.
    point : tuple or numpy.ndarray
        A tuple or array containing the correlators `e1` and `e2`. These are fixed
        values used in the evaluation of the partial derivatives.

    Returns:
    --------
    numpy.ndarray
        A 2x2 Jacobian matrix of the form:
        \[
        \begin{bmatrix}
        \frac{\partial F}{\partial a} & \frac{\partial F}{\partial c} \\
        \frac{\partial G}{\partial a} & \frac{\partial G}{\partial c}
        \end{bmatrix}
        \]
        where the partial derivatives are evaluated at `(a, c, e1, e2)`.

    Notes:
    ------
    - The functions `Dfa`, `Dfc`, `Dga`, and `Dgc` are defined in the polynomials.py
      file. These functions compute the partial derivatives of `F` and `G` with
      respect to `a` and `c`.
    - Providing the Jacobian matrix can significantly improve the performance and
      accuracy of numerical solvers like `scipy.optimize.root`.

    Examples:
    ---------
    >>> from scipy.optimize import root
    >>> from triangle_inequalities import fun, jac, e3fun
    >>> e1, e2 = 1/3, -5/27
    >>> a, c = root(fun, (0.25, 0.25), args=((e1, e2), ), jac=jac).x
    >>> print(f'a = {a}, c = {c}')
    >>> print(f'E3 = {e3fun(a, c, e1, e2)}')
    """
    return np.array([[Dfx(*x, *point), Dfz(*x, *point)],
                     [Dgx(*x, *point), Dgz(*x, *point)]])


# -------------------------------------------------------- Network inequalities
def GHZ(e1, e2, e3, flip=False):
    """
    Computes the inequality for the GHZ surface involving the correlators `e1`, `e2`,
    and `e3`. If `GHZ >= 0`, the behavior is not in the GHZ region. Optionally, the
    behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `GHZ` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `GHZ`. If `GHZ >= 0`, the behavior is not in
        the GHZ region; otherwise, the behavior might be nonlocal, in case it also
        violates the flipped version of the inequality.

    Examples:
    ---------
    >>> from triangle_inequalities import GHZ
    >>> e1, e2, e3 = 0, 1, 0
    >>> violated = [GHZ(e1, e2, e3, flip=flip) < 0 for flip in (False, True)]
    >>> non_local = all(violated)
    >>> print(f'Is GHZ nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    ghz = 8*(1+e1)**3*(11 + 37*e1 + 28*e1**2 - 4*e1**3 + 8*e2 + 13*e1*e2
                       + e2**2 - 11*e3 - 18*e1*e3 - 3*e2*e3 + 2*e3**2) \
        - (3 + 5*e1 + e2 - e3)**4 + np.sqrt((3 + 5*e1 + e2 - e3)**2 
                                            - 8*(1+e1)**3)\
            *((3 + 5*e1 + e2 - e3)**3 - 4*(1+e1)**3*(7 + 11*e1 + e2 - 3*e3))
    return np.where((3 + 5*e1 + e2 - e3)**2 >= 8*(1 + e1)**3, ghz, np.inf)


def W1(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W green surface involving the correlators `e1`,
    `e2`, and `e3`. If `W1 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W1` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W1`. If `W1 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    w1 = -4*e1**2 + 4*e1**3 + (1 + e2 + np.sqrt((1+e2)**2 - 4*e1**2))\
        *(1 - e1 + e2 + e3 - 2*e1*e2)
    return w1


def W2(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W purple surface involving the correlators `e1`,
    `e2`, and `e3`. If `W2 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W2` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W2`. If `W2 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    w2 = 8*e1**5 - 15*e1**4 - 16*e1**3*e2 + 6*e1**3*e3 + 22*e1**3 + 16*e1**2*e2**2 - 8*e1**2*e2*e3 - 2*e1**2*e2 + e1**2*e3**2 - 12*e1**2*e3 - 20*e1**2 - 6*e1*e2**2 + 10*e1*e2*e3 + 12*e1*e2 - 2*e1*e3**2 + 10*e1*e3 + 10*e1 - e2**2 - 8*e2*e3 - 6*e2 + 2*e3**2 - 1
    return np.where(e1 >= 1/3, w2, -np.inf)


def W3(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W red surface involving the correlators `e1`,
    `e2`, and `e3`. If `W3 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W3` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W3`. If `W3 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    valid = ((2048*e1**12 - 1024*e1**11*e2 - 3832*e1**11 + 2196*e1**10*e2 + 5508*e1**10 + 1062*e1**9*e2**2 + 2796*e1**9*e2 + 12198*e1**9 - 729*e1**8*e2**3 - 543*e1**8*e2**2 - 23427*e1**8*e2 - 20061*e1**8 - 1404*e1**7*e2**3 + 12636*e1**7*e2**2 + 29532*e1**7*e2 + 8084*e1**7 + 1368*e1**6*e2**3 - 2112*e1**6*e2**2 - 27680*e1**6*e2 - 1416*e1**6 - 2484*e1**5*e2**4 - 156*e1**5*e2**3 + 43568*e1**5*e2**2 + 2324*e1**5*e2 + 9212*e1**5 - 3456*e1**4*e2**4 - 18162*e1**4*e2**3 + 19706*e1**4*e2**2 - 25430*e1**4*e2 - 17250*e1**4 - 2664*e1**3*e2**4 - 17620*e1**3*e2**3 + 33876*e1**3*e2**2 + 44052*e1**3*e2 + 9340*e1**3 + 432*e1**2*e2**5 + 848*e1**2*e2**4 - 20128*e1**2*e2**3 - 31704*e1**2*e2**2 - 12700*e1**2*e2 - 500*e1**2 + 2268*e1*e2**4 + 5100*e1*e2**3 + 2682*e1*e2**2 - 624*e1*e2 - 506*e1 + 243*e2**3 + 477*e2**2 + 369*e2 + 71 <= 0)
             & (4*e1**4 - 8*e1**3 + 4*e1**2*e2 + 12*e1**2 + 2*e1*e2**2 + 4*e1*e2
                + 2*e1 - e2**2 - 2*e2 - 1 >= 0)
             & (e3 >= 6 - 3*np.sqrt(5)))
    w3 = np.where(valid,
                  216*e1**5 - 108*e1**4*e2 - 162*e1**4 - 54*e1**3*e2 + 270*e1**3*e3 - 54*e1**3 + 162*e1**2*e2**2 - 216*e1**2*e2*e3 + 351*e1**2*e2 + 54*e1**2*e3**2 - 270*e1**2*e3 + 189*e1**2 - 54*e1*e2**3 + 54*e1*e2**2*e3 - 216*e1*e2**2 - 18*e1*e2*e3**2 + 162*e1*e2*e3 - 270*e1*e2 + 2*e1*e3**3 + 18*e1*e3**2 + 108*e1*e3 - 108*e1 + 27*e2**3 + 81*e2**2 - 9*e2*e3**2 + 81*e2 + 2*e3**3 - 9*e3**2 + 27, 
                  -np.inf)
    return w3


def W4(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W yellow surface involving the correlators `e1`,
    `e2`, and `e3`. If `W4 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip` 
        function before evaluating `W4` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W4`. If `W4 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    w4 = 256*e1**8 - 256*e1**7*e2 - 882*e1**7 + 64*e1**6*e2**2 + 504*e1**6*e2 + 88*e1**6*e3 + 1013*e1**6 + 212*e1**5*e2**2 - 232*e1**5*e2*e3 + 161*e1**5*e2 + 36*e1**5*e3**2 - 299*e1**5*e3 - 252*e1**5 - 184*e1**4*e2**3 + 168*e1**4*e2**2*e3 - 766*e1**4*e2**2 - 24*e1**4*e2*e3**2 + 476*e1**4*e2*e3 - 919*e1**4*e2 - 8*e1**4*e3**3 - 62*e1**4*e3**2 + 409*e1**4*e3 - 345*e1**4 + 30*e1**3*e2**4 - 40*e1**3*e2**3*e3 + 258*e1**3*e2**3 + 4*e1**3*e2**2*e3**2 - 138*e1**3*e2**2*e3 + 432*e1**3*e2**2 + 8*e1**3*e2*e3**3 - 66*e1**3*e2*e3**2 - 192*e1**3*e2*e3 + 530*e1**3*e2 - 2*e1**3*e3**4 + 42*e1**3*e3**3 + 16*e1**3*e3**2 - 278*e1**3*e3 + 270*e1**3 - 7*e1**2*e2**4 - 36*e1**2*e2**3*e3 + 118*e1**2*e2**3 + 86*e1**2*e2**2*e3**2 - 230*e1**2*e2**2*e3 + 364*e1**2*e2**2 - 52*e1**2*e2*e3**3 + 154*e1**2*e2*e3**2 - 152*e1**2*e2*e3 + 98*e1**2*e2 + 9*e1**2*e3**4 - 42*e1**2*e3**3 + 12*e1**2*e3**2 + 82*e1**2*e3 - 61*e1**2 - 3*e1*e2**5 + 13*e1*e2**4*e3 - 52*e1*e2**4 - 22*e1*e2**3*e3**2 + 112*e1*e2**3*e3 - 290*e1*e2**3 + 18*e1*e2**2*e3**3 - 88*e1*e2**2*e3**2 + 250*e1*e2**2*e3 - 404*e1*e2**2 - 7*e1*e2*e3**4 + 32*e1*e2*e3**3 - 62*e1*e2*e3**2 + 136*e1*e2*e3 - 147*e1*e2 + e1*e3**5 - 4*e1*e3**4 + 6*e1*e3**3 - 4*e1*e3**2 + e1*e3 + e2**5 - 3*e2**4*e3 + 29*e2**4 + 2*e2**3*e3**2 - 36*e2**3*e3 + 98*e2**3 + 2*e2**2*e3**3 - 2*e2**2*e3**2 - 50*e2**2*e3 + 98*e2**2 - 3*e2*e3**4 + 12*e2*e3**3 - 2*e2*e3**2 - 36*e2*e3 + 29*e2 + e3**5 - 3*e3**4 + 2*e3**3 + 2*e3**2 - 3*e3 + 1
    return w4


def W5(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W cyan surface involving the correlators `e1`,
    `e2`, and `e3`. If `W5 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W5` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W5`. If `W5 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3_W5 = np.full_like(e1.flatten(), np.inf)
    initial_guesses = np.array([[0.28, 0.44], 
                                [0.36, 0.46], 
                                [0.32, 0.3], 
                                [0.3, 0.5], 
                                [0.4, 0.44]])
    
    # intersection with D0 W Wflip plane
    e = np.linspace(1/2, 1/3, 40)
    border1 = np.vstack((e, (2*np.sqrt(2*(3*e-1))*(1-3*e) + 36*e - 17)/27)).T
    params1 = np.vstack(((2 + np.sqrt(2*(3*e-1)))/6, (1 - np.sqrt(2*(3*e-1)))/3)).T
    
    # intersection with red surface
    e = np.linspace(1/3, 1, 40)
    a = np.empty_like(e)
    for i, e_ in enumerate(e):
        a[i] = Polynomial([4*e_**2*(1 + e_)**2, 
                           -5*e_ - 16*e_**2 - 50*e_**3 - 24*e_**4 - e_**5, 
                           1 + 12*e_ + 34*e_**2 + 84*e_**3 + 45*e_**4, 
                           -12*e_ - 24*e_**2 - 60*e_**3, 
                           16*e_**2]).roots()[0].real
    border2 = np.vstack(((1-3*a)*e/(a-e), 
                         (-a + 3*e - 6*a*e + 4*a**3*e - 4*a**2*e**2)/(a - e))).T
    params2 = np.vstack((a, a)).T
    border = np.vstack((border1, border2))
    params = np.vstack((params1, params2))
    
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        
        if (e1_ > 1/2 or e1_ < np.sqrt(5)-2 
                or e2_ > 9-4*np.sqrt(5) or e2_ < -5/27 
                or e2_ < (-5 + 2*(3*e1_-1)*(6 - np.sqrt(2*(3*e1_-1))))/27):
            continue
        
        distances_square = np.sum((border - np.array([e1_, e2_]))**2, axis=1)
        initial_point = border[np.argmin(distances_square)]
        initial_params = params[np.argmin(distances_square)]
        displacement = np.array([e1_, e2_]) - initial_point
        distance = np.sqrt(np.sum(displacement**2))
        direction = displacement/distance
        step = 0.01
        alphas = np.arange(0, distance, step)
        x0 = initial_params
        
        for alpha in alphas:
            point = initial_point + alpha*direction
            solution = root(fun, x0, args=point, jac=jac)
            x0 = solution.x
        
        solution = root(fun, x0, args=((e1_, e2_), ), jac=jac)
        x, z = solution.x
        y = yfun(x, z, e1_, e2_)
        t = tfun(x, z, e1_, e2_)
        u = ufun(x, z, e1_, e2_)
        v = vfun(x, z, e1_, e2_)
        w = wfun(x, z, e1_, e2_)
        k = kfun(x, z, e1_, e2_)
        e3_ = e3fun(x, z, e1_, e2_)
        if (x>=0 and y>=0 and x+y<=1 and z>=0 and t>=0 and z+t<=1 and u>=0
                and v>=0 and u+v<=1 and w>=0 and w<=1
                and np.isclose(k, 1, rtol=1e-3) and e3_ < e3_W5[i]):
            e3_W5[i] = e3fun(x, z, e1_, e2_)
        
        for x0 in initial_guesses:
            solution = root(fun, x0, args=((e1_, e2_), ), method='hybr', jac=jac)
            x, z = solution.x
            y = yfun(x, z, e1_, e2_)
            t = tfun(x, z, e1_, e2_)
            u = ufun(x, z, e1_, e2_)
            v = vfun(x, z, e1_, e2_)
            w = wfun(x, z, e1_, e2_)
            k = kfun(x, z, e1_, e2_)
            e3_ = e3fun(x, z, e1_, e2_)
            if (x>=0 and y>=0 and x+y<=1 and z>=0 and t>=0 and z+t<=1 and u>=0
                    and v>=0 and u+v<=1 and w>=0 and w<=1
                    and np.isclose(k, 1, rtol=1e-3) and e3_ < e3_W5[i]):
                e3_W5[i] = e3fun(x, z, e1_, e2_)

    w5 = e3 - e3_W5
    return w5.reshape(e1.shape)


# ---------------------------------------------- Tests for triangle nonlocality
def W_test(e1, e2, e3, flip=False):
    """
    Tests whether all the W inequalities are violated for a given set of correlators.

    The function evaluates the W inequalities `W1`, `W2`, `W3`, `W4`, and `W5` for the
    input correlators `e1`, `e2`, and `e3`. If all inequalities are violated (i.e.,
    all are negative), the function returns `True`, indicating that the behavior is in
    the W region. Otherwise, it returns `False`.

    By using the `flip` flag, this function can be used to test membership in the
    flipped W region.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The first correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The second correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The third correlator. Can be a single float or a numpy array of floats.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the
        `bit_flip` function before evaluating the inequalities (default is `False`).

    Returns:
    --------
    bool or numpy.ndarray
        - If the inputs are floats, returns `True` if all W inequalities are violated,
          and `False` otherwise.
        - If the inputs are numpy arrays, returns a boolean array of the same shape,
          where each element indicates whether all W inequalities are violated for the
          corresponding set of correlators.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> is_in_W_region = W_test(e1, e2, e3)
    >>> print("Is the behavior in the W region?", is_in_W_region)

    >>> # Example with numpy arrays
    >>> e1 = np.array([1/3, -1/3])
    >>> e2 = np.array([-1/3, -1/3])
    >>> e3 = np.array([-1, 1])
    >>> is_in_W_region = W_test(e1, e2, e3)
    >>> print("Are the behaviors in the W region?", is_in_W_region)

    >>> # Using the flip option
    >>> is_in_flipped_W_region = W_test(e1, e2, e3, flip=True)
    >>> print("Are the behaviors in the flipped W region?", is_in_flipped_W_region)
    """
    return ((W1(e1, e2, e3, flip) < 0) & (W2(e1, e2, e3, flip) < 0)
            & (W3(e1, e2, e3, flip) < 0) & (W4(e1, e2, e3, flip) < 0)
            & (W5(e1, e2, e3, flip) < 0))


def GHZ_test(e1, e2, e3):
    """
        Tests whether the GHZ inequality and its bit-flipped version are violated
        for a given set of correlators.

        The function evaluates the inequalities for the input correlators `e1`,
        `e2`, and `e3`. If all inequalities are violated (i.e., all are negative),
        the function returns `True`, indicating that the behavior is in the
        GHZ region. Otherwise, it returns `False`.

        Parameters:
        -----------
        e1 : float or numpy.ndarray
            The first correlator. Can be a single float or a numpy array of floats.
        e2 : float or numpy.ndarray
            The second correlator. Can be a single float or a numpy array of floats.
        e3 : float or numpy.ndarray
            The third correlator. Can be a single float or a numpy array of floats.

        Returns:
        --------
        bool or numpy.ndarray
            - If the inputs are floats, returns `True` if all GHZ inequalities
              are violated, and `False` otherwise.
            - If the inputs are numpy arrays, returns a boolean array of the same shape,
              where each element indicates whether all GHZ inequalities are violated
              for the corresponding set of correlators.

        Examples:
        ---------
        >>> # Example with floats
        >>> e1, e2, e3 = 0, 1, 0
        >>> is_in_GHZ_region = GHZ_test(e1, e2, e3)
        >>> print("Is the behavior in the GHZ region?", is_in_GHZ_region)

        >>> # Example with numpy arrays
        >>> e1 = np.array([0, 1, 1/3])
        >>> e2 = np.array([1, 1, -1/3])
        >>> e3 = np.array([0, 1, -1])
        >>> is_in_GHZ_region = GHZ_test(e1, e2, e3)
        >>> print("Are the behaviors in the GHZ region?", is_in_GHZ_region)
        """
    return (GHZ(e1, e2, e3) < 0) & (GHZ(e1, e2, e3, flip=True) < 0)


def nonlocal_test(e1, e2, e3):
    """
    Tests membership to the GHZ, W, and flipped W regions for a given set of correlators.

    The function evaluates whether the behavior defined by the correlators `e1`, `e2`,
    and `e3` exhibits GHZ, W, or flipped W type of nonlocality. It returns:
    - `0` if the behavior is local (i.e., it does not belong to any of the nonlocal regions).
    - `1` if the behavior exhibits GHZ-type nonlocality.
    - `2` if the behavior exhibits W-type nonlocality.
    - `3` if the behavior exhibits flipped W-type nonlocality.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The first correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The second correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The third correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    int or numpy.ndarray
        - If the inputs are floats, returns an integer:
            - `0`: Behavior is local.
            - `1`: Behavior exhibits GHZ-type nonlocality.
            - `2`: Behavior exhibits W-type nonlocality.
            - `3`: Behavior exhibits flipped W-type nonlocality.
        - If the inputs are numpy arrays, returns an array of integers with the same
          shape, where each element corresponds to the result for the corresponding
          set of correlators.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> result = nonlocal_test(e1, e2, e3)
    >>> print("Nonlocality type:", result)

    >>> # Example with numpy arrays
    >>> e1 = np.array([1, 0, 1/3, -1/3])
    >>> e2 = np.array([1, 1, -1/3, -1/3])
    >>> e3 = np.array([1, 0, -1, 1])
    >>> result = nonlocal_test(e1, e2, e3)
    >>> print("Nonlocality types:", result)
    """
    ghz = GHZ_test(e1, e2, e3)
    w = W_test(e1, e2, e3)
    w_flip = W_test(e1, e2, e3, flip=True)
    return (ghz) + (w)*2 + (w_flip)*3


# ------------------------------ Generate points on boundary (given projection)
def GHZ_point(e1, e3):
    """
    Computes the values of `e2` for points on the GHZ surface, given the
    correlators `e1` and `e3`.

    The function evaluates a mathematical expression to compute `e2` for points lying
    on the GHZ surface. It also ensures that the computed `e2` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The three-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e2` values corresponding to points on the GHZ surface.
        If the computed `e2` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e2` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e3 = 0, 0
    >>> e3 = GHZ_point(e1, e3)
    >>> print(f"Point on green surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e3 = np.linspace(-1, 1, 30)
    >>> e1, e3 = np.meshgrid(e1, e3)
    >>> e2 = GHZ_point(e1, e3)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> GHZ_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e3 = np.atleast_1d(np.array(e1), np.array(e3))
    e2 = np.full_like(e1.flatten(), np.inf, float)
    for i, (e1_, e3_) in enumerate(zip(e1.flatten(), e3.flatten())):
        roots = np.roots([4, 0, 0, -8 - 12 * e1_ + 4 * e3_, 3 + 6 * e1_ + 3 * e1_ ** 2])
        b = roots[~np.iscomplex(roots)].real
        a = (1 - 2 * b ** 2 + e1_) / (4 * b)
        valid = (a >= 0) & (b >= 0) & (a + b <= 1)
        if valid.sum() >= 1:
            a, b = a[valid][0], b[valid][0]
            print(f'{a=}, {b=}, {e1_=}, {e3_=}, {e2[i]=}')
            e2[i] = 1 - 8*a*b + 4*a**2*b - 4*b**2 + 12*a*b**2 + 4*b**3
            print(f'{a=}, {b=}, {e1_=}, {e3_=}, {e2[i]=}')

    return e2.reshape(e1.shape)


def W1_point(e1, e2):
    """
    Computes the values of `e3` for points on the W green surface, given the correlators
    `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W green surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W green surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 0, 0
    >>> e3 = W1_point(e1, e2)
    >>> print(f"Point on green surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W1_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> green_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e2 = np.atleast_1d(e1, e2)
    e3 = -2*(2*e1 - e2 - 1)*(4*e1**3 - 4*e1**2 - e1*e2 - e1 + e2**2 
                             + 2*e2 + (e1 - e2 - 1)*np.sqrt(-4*e1**2 + e2**2 
                                                            + 2*e2 + 1) + 1)\
        /(2*e1 - e2 + np.sqrt(-4*e1**2 + e2**2 + 2*e2 + 1) - 1)**2
    e3[(e1 == 0) & (e2 != 0)] = np.inf
    e3[(e1 == 0) & (e2 == 0)] = -1
    valid = valid_distribution(e1, e2, e3)
    return np.where(valid, e3, np.inf)


def W2_point(e1, e2):
    """
        Computes the values of `e3` for points on the W purple surface, given the
        correlators `e1` and `e2`.

        The function evaluates a mathematical expression to compute `e3` for points lying
        on the W purple surface. It also ensures that the computed `e3` values correspond
        to valid probability distributions by checking the `valid_distribution` function.
        If the distribution is invalid, the function returns `np.inf` for those points.

        Parameters:
        -----------
        e1 : float or numpy.ndarray
            The one-body correlator. Can be a single float or a numpy array of floats.
        e2 : float or numpy.ndarray
            The two-body correlator. Can be a single float or a numpy array of floats.

        Returns:
        --------
        numpy.ndarray
            An array of `e3` values corresponding to points on the W purple surface.
            If the computed `e3` values do not correspond to a valid probability
            distribution, the function returns `np.inf` for those points.

        Notes:
        ------
        - The `valid_distribution` function is used to check if the computed `e3` values
          correspond to valid probability distributions.
        - This function is not guaranteed to work for points on the surface boundary.

        Examples:
        ---------
        >>> # Example with floats
        >>> e1, e2 = 1/3, 1/9
        >>> e3 = W2_point(e1, e2)
        >>> print(f"Point on purple surface: {e1, e2, e3}")

        >>> # Example with numpy arrays
        >>> e1 = np.linspace(-1, 1, 30)
        >>> e2 = np.linspace(-1/3, 1, 20)
        >>> e1, e2 = np.meshgrid(e1, e2)
        >>> e3 = W2_point(e1, e2)
        >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
        >>> purple_surface = np.vstack((e1, e2, e3)).T
        """
    e1, e2 = np.atleast_1d(np.array(e1), np.array(e2))
    e3 = (68*e1**6 + 11*e1**5*e2 - 411*e1**5 - 4*e1**4*e2**2 - 123*e1**4*e2 + 795*e1**4 + 141*e1**3*e2**2 + 495*e1**3*e2 - 784*e1**3 - 557*e1**2*e2**2 - 407*e1**2*e2 + 364*e1**2 + 543*e1*e2**2 + 18*e1*e2 - 13*e1 - 99*e2**2 - 34*e2 + np.sqrt(2)*np.sqrt(1 - e1)*(-8*e1**6 - 2*e1**5*e2 + 140*e1**5 + 24*e1**4*e2 - 378*e1**4 - 25*e1**3*e2**2 - 203*e1**3*e2 + 430*e1**3 + 228*e1**2*e2**2 + 290*e1**2*e2 - 252*e1**2 - 349*e1*e2**2 - e1*e2 + 10*e1 + 70*e2**2 + 24*e2 + 2) - 3)/(-4*e1**5 - e1**4*e2 + 49*e1**4 + 33*e1**3*e2 - 109*e1**3 - 131*e1**2*e2 + 111*e1**2 + 131*e1*e2 - 51*e1 - 24*e2 + np.sqrt(2)*np.sqrt(1 - e1)*(-14*e1**4 - 6*e1**3*e2 + 52*e1**3 + 53*e1**2*e2 - 59*e1**2 - 84*e1*e2 + 38*e1 + 17*e2 + 3) - 4)
    valid = valid_distribution(e1, e2, e3)
    return np.where(valid, e3, np.inf)


def W3_point(e1, e2):
    """
    Computes the values of `e3` for points on the W red surface, given the
    correlators `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W red surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W red surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 0.2, 0.2
    >>> e3 = W3_point(e1, e2)
    >>> print(f"Point on red surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W3_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> red_surface = np.vstack((e1, e2, e3)).T
    """
    a = (1/2)*(2*e1 - e2 + ((-2*e1 + e2 + 1)*(e1**(3/2) + np.sqrt(e1) + np.sqrt(e1**3 + 2*e1**2 - e1 + e2 + 1)))**(2/3) - 1)/(np.sqrt(e1)*((-2*e1 + e2 + 1)*(e1**(3/2) + np.sqrt(e1) + np.sqrt(e1**3 + 2*e1**2 - e1 + e2 + 1)))**(1/3))
    b = -a*(4*a**2 + 4*a*e1 - 4*a - 2*e1 + e2 + 1)/(4*a**2 + 2*e1 - e2 - 1)
    e = (1 - 2*e1 + e2)/4/a**2
    f = (-1 + 4*a**2 + 2*e1 -e2)/8/a**2
    s = (1/4)*(-8*a**3*e1 + 20*a**2*e1 - 4*a**2*e2 - 8*a**2 - 12*a*e1 + 6*a*e2 + 6*a + 2*e1**2 - e1*e2 + e1 - e2 - 1)/(a**2*(2*e1 - e2 - 1))
    t = (1/4)*(4*a**2 + 2*e1 - e2 - 1)**2*(8*a**3*e1 - 4*a**2*e1 - 4*a**2*e2 - 4*a*e1 + 2*a*e2 + 2*a + 2*e1**2 - e1*e2 + e1 - e2 - 1)/(a**2*(2*e1 - e2 - 1)*(4*a**2 + 4*a*e1 - 4*a - 2*e1 + e2 + 1)**2)
    
    x = -16*e1**7 + 8*e1**6*e2 - 56*e1**6 + 32*e1**5*e2 - 144*e1**5 + 128*e1**4*e2 - 96*e1**4 - 20*e1**3*e2**2 + 152*e1**3*e2 + 84*e1**3 - 40*e1**2*e2**2 - 4*e1**2*e2 + 36*e1**2 - 14*e1*e2**2 - 28*e1*e2 - 14*e1 - e2**3 - 3*e2**2 - 3*e2 + 8*np.sqrt(e1*(e1 + 1)**2*(2*e1 - e2 - 1)**2*(e1**3 + 2*e1**2 - e1 + e2 + 1)**3) - 1
    e3 = (3/4)*(32*e1**4 - 16*e1**3*e2 + 48*e1**3 - 32*e1**2*e2 - 12*e1**2*x**(1/3) + 8*e1**2 + 4*e1*e2*x**(1/3) - 24*e1*e2 - 4*e1*x**(1/3) - 24*e1 + 2*e2**2 + 2*e2*x**(1/3) + 4*e2 + 2*x**(2/3) + 2*x**(1/3) + 2)/(x**(1/3)*(e1 + 1))
    return np.where((a>=0) & (b>=0) & (a+b<=1) & (e>=0) & (f>=0) & (e+f<=1) 
                    & (s>=0) & (s<=1) & (t>=0) & (t<=1), 
                    e3, np.inf)


def W4_point(e1, e2):
    """
    Computes the values of `e3` for points on the W yellow surface, given the
    correlators `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W yellow surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W yellow surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 0.4, 0
    >>> e3 = W4_point(e1, e2)
    >>> print(f"Point on yellow surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W4_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> yellow_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3 = np.full_like(e1.flatten(), np.inf, float)
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        poly = Polynomial([4*e1_**3 - 4*e1_**2*e2_ + e1_*e2_**2 - 2*e1_*e2_ - 3*e1_ + e2_**2 + 2*e2_ + 1, 
                           e1_**5 - 5*e1_**4 + 3*e1_**3*e2_ - 13*e1_**3 + 13*e1_**2*e2_ - 3*e1_**2 - 3*e1_*e2_**2 + 11*e1_*e2_ + 17*e1_ - 5*e2_**2 - 11*e2_ - 5, 
                           8*e1_**4 - 4*e1_**3*e2_ + 20*e1_**3 - 16*e1_**2*e2_ + 16*e1_**2 + 3*e1_*e2_**2 - 26*e1_*e2_ - 41*e1_ + 9*e2_**2 + 22*e2_ + 9, 
                           -4*e1_**4 - 16*e1_**3 + 8*e1_**2*e2_ - 32*e1_**2 - e1_*e2_**2 + 30*e1_*e2_ + 51*e1_ - 7*e2_**2 - 22*e2_ - 7, 
                           32*e1_**2 - 16*e1_*e2_ - 32*e1_ + 2*e2_**2 + 12*e2_ + 2, 
                           4*e1_**3 - 12*e1_**2 + 4*e1_*e2_ + 8*e1_ - 4*e2_])
        c = poly.roots()
        c = c[~np.iscomplex(c)].real
        a = (2*e1_ - e2_ - 1)*(2*c**2 - 2*c*e1_ - 2*c + e1_ + 1)/(8*c**2*e1_ - 4*c*e1_**2 - 8*c*e1_ + 2*c*e2_ + 2*c + 4*e1_ - 2*e2_ - 2)
        b = -a*(4*a**2*c*e1_ + 4*a**2*c - 4*a*c**2*e1_ + 4*a*c**2 - 8*a*c - 4*a*e1_ + 2*a*e2_ + 2*a + 4*c**2*e1_ - 4*c**2 - 4*c*e1_**2 + 4*c + 2*e1_**2 - e1_*e2_ + e1_ - e2_ - 1)/(4*a**2*c*e1_ + 4*a**2*c - 4*a*c**2*e1_ + 4*a*c**2 - 4*a*c*e1_ - 4*a*c + 4*c*e1_ - 2*c*e2_ - 2*c - 2*e1_**2 + e1_*e2_ - e1_ + e2_ + 1)
        d = -c*(4*a**2*c*e1_ - 4*a**2*c - 4*a**2*e1_ + 4*a**2 - 4*a*c**2*e1_ - 4*a*c**2 + 8*a*c + 4*a*e1_**2 - 4*a + 4*c*e1_ - 2*c*e2_ - 2*c - 2*e1_**2 + e1_*e2_ - e1_ + e2_ + 1)/(4*a**2*c*e1_ - 4*a**2*c - 4*a*c**2*e1_ - 4*a*c**2 + 4*a*c*e1_ + 4*a*c - 4*a*e1_ + 2*a*e2_ + 2*a + 2*e1_**2 - e1_*e2_ + e1_ - e2_ - 1)
        e = (1/4)*(-2*e1_ + e2_ + 1)/(a*c)
        f = (1/8)*(4*a**2*c*e1_ + 4*a**2*c - 4*a*c**2*e1_ + 4*a*c**2 - 4*a*c*e1_ - 4*a*c + 4*c*e1_ - 2*c*e2_ - 2*c - 2*e1_**2 + e1_*e2_ - e1_ + e2_ + 1)/(a*c*(a + c - e1_ - 1))
        s = (1/4)*(16*a**2*c**2*e1_ - 4*a**2*c*e1_**2 - 24*a**2*c*e1_ + 4*a**2*c*e2_ + 8*a**2*c + 8*a**2*e1_ - 4*a**2*e2_ - 4*a**2 - 4*a*c**2*e1_**2 - 24*a*c**2*e1_ + 4*a*c**2*e2_ + 8*a*c**2 + 20*a*c*e1_**2 - 4*a*c*e1_*e2_ + 20*a*c*e1_ - 8*a*c*e2_ - 12*a*c - 8*a*e1_**2 + 4*a*e1_*e2_ - 4*a*e1_ + 4*a*e2_ + 4*a + 8*c**2*e1_ - 4*c**2*e2_ - 4*c**2 - 8*c*e1_**2 + 4*c*e1_*e2_ - 4*c*e1_ + 4*c*e2_ + 4*c + 2*e1_**3 - e1_**2*e2_ + 3*e1_**2 - 2*e1_*e2_ - e2_ - 1)/(a*c*(2*e1_ - e2_ - 1)*(-a - c + e1_ + 1))
        valid = ((a >= 0) & (b >= 0) & (a + b <= 1) & (c >= 0) & (d>= 0) 
                 & (c + d <= 1) & (e >= 0) & (f.real >= 0) & (e + f <= 1) 
                 & (s >= 0) & (s <= 1))
        if valid.sum() != 0:
            e3[i] = np.min((1/2)*(16*a**2*c**2*e1_ - 4*a**2*c*e1_**2 - 22*a**2*c*e1_ + 6*a**2*c*e2_ + 6*a**2*c + 8*a**2*e1_ - 4*a**2*e2_ - 4*a**2 - 4*a*c**2*e1_**2 - 22*a*c**2*e1_ + 6*a*c**2*e2_ + 6*a*c**2 + 18*a*c*e1_**2 - 6*a*c*e1_*e2_ + 20*a*c*e1_ - 10*a*c*e2_ - 10*a*c - 8*a*e1_**2 + 4*a*e1_*e2_ - 4*a*e1_ + 4*a*e2_ + 4*a + 8*c**2*e1_ - 4*c**2*e2_ - 4*c**2 - 8*c*e1_**2 + 4*c*e1_*e2_ - 4*c*e1_ + 4*c*e2_ + 4*c + 2*e1_**3 - e1_**2*e2_ + 3*e1_**2 - 2*e1_*e2_ - e2_ - 1)/(a*c*(a + c - e1_ - 1)))
    return e3.reshape(e1.shape)


def W5_point(e1, e2):
    """
    Computes the values of `e3` for points on the W cyan surface, given the
    correlators `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W cyan surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W cyan surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 1/3, -5/27
    >>> e3 = W5_point(e1, e2)
    >>> print(f"Point on cyan surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W5_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> cyan_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3 = np.full_like(e1.flatten(), np.inf, float)
    
    # intersection with D0 W Wflip plane
    e = np.linspace(1/2, 1/3, 40)
    border1 = np.vstack((e, (2*np.sqrt(2*(3*e-1))*(1-3*e) + 36*e - 17)/27)).T
    params1 = np.vstack(((2 + np.sqrt(2*(3*e-1)))/6, (1 - np.sqrt(2*(3*e-1)))/3)).T
    
    # intersection with red surface
    e = np.linspace(1/3, 1, 40)
    a = np.empty_like(e)
    for i, e_ in enumerate(e):
        a[i] = Polynomial([4*e_**2*(1 + e_)**2, 
                           -5*e_ - 16*e_**2 - 50*e_**3 - 24*e_**4 - e_**5, 
                           1 + 12*e_ + 34*e_**2 + 84*e_**3 + 45*e_**4, 
                           -12*e_ - 24*e_**2 - 60*e_**3, 
                           16*e_**2]).roots()[0].real
    border2 = np.vstack(((1-3*a)*e/(a-e), 
                         (-a + 3*e - 6*a*e + 4*a**3*e - 4*a**2*e**2)/(a - e))).T
    params2 = np.vstack((a, a)).T
    border = np.vstack((border1, border2))
    params = np.vstack((params1, params2))
    
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        
        if (e1_ > 1/2 or e1_ < np.sqrt(5)-2 
                or e2_ > 9-4*np.sqrt(5) or e2_ < -5/27 
                or e2_ < (-5 + 2*(3*e1_-1)*(6 - np.sqrt(2*(3*e1_-1))))/27):
            continue
        
        distances_square = np.sum((border - np.array([e1_, e2_]))**2, axis=1)
        initial_point = border[np.argmin(distances_square)]
        initial_params = params[np.argmin(distances_square)]
        displacement = np.array([e1_, e2_]) - initial_point
        distance = np.sqrt(np.sum(displacement**2))
        direction = displacement/distance
        step = 0.003
        alphas = np.arange(0, distance, step)
        x0 = initial_params
        
        for alpha in alphas:
            point = initial_point + alpha*direction
            solution = root(fun, x0, args=point, jac=jac)
            x0 = solution.x
        
        solution = root(fun, x0, args=((e1_, e2_), ), jac=jac)
        x, z = solution.x
        y = yfun(x, z, e1_, e2_)
        t = tfun(x, z, e1_, e2_)
        u = ufun(x, z, e1_, e2_)
        v = vfun(x, z, e1_, e2_)
        w = wfun(x, z, e1_, e2_)
        k = kfun(x, z, e1_, e2_)
        e3_ = e3fun(x, z, e1_, e2_)
        if (x>=0 and y>=0 and x+y<=1 and z>=0 and t>=0 and z+t<=1 and u>=0
                and v>=0 and u+v<=1 and w>=0 and w<=1
                and np.isclose(k, 1, rtol=1e-3) and e3_ < e3[i]):
            e3[i] = e3fun(x, z, e1_, e2_)
    return e3


def W5_point_fixed_initial_guesses(e1, e2):
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3 = np.full_like(e1.flatten(), np.inf)
    initial_guesses = np.array([[0.28, 0.44], 
                                [0.36, 0.46], 
                                [0.32, 0.3], 
                                [0.3, 0.5], 
                                [0.4, 0.44]])
    
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        
        if e1_ > 1/2 or e1_ < np.sqrt(5)-2:
            continue
        
        for x0 in initial_guesses:
            solution = root(fun, x0, args=((e1_, e2_), ), method='hybr', jac=jac)
            x, z = solution.x
            y = yfun(x, z, e1_, e2_)
            t = tfun(x, z, e1_, e2_)
            u = ufun(x, z, e1_, e2_)
            v = vfun(x, z, e1_, e2_)
            w = wfun(x, z, e1_, e2_)
            k = kfun(x, z, e1_, e2_)
            e3_ = e3fun(x, z, e1_, e2_)
            if (x>=0 and y>=0 and x+y<=1 and z>=0 and t>=0 and z+t<=1 and u>=0
                    and v>=0 and u+v<=1 and w>=0 and w<=1
                    and np.isclose(t, 1, rtol=1e-3) and e3_ < e3[i]):
                e3[i] = e3fun(x, z, e1_, e2_)
    return e3


# ---------------------------------------- Generate points on boundary (random)
def choose_GHZ_point():
    """
    Randomly selects a point on the GHZ surface.

    The function generates random values for the model parameters `x` and `y`
    and computes the corresponding correlators `e1`, `e2`, and `e3`. It then
    checks if the point satisfies the flipped version of the GHZ inequality. If
    it does, the function recursively calls itself to generate a new point
    in order to take a point on the local boundary.

    Returns:
    --------
    tuple
        A tuple containing the correlators `(e1, e2, e3)` of a randomly selected point
        on the GHZ boundary.

    Examples:
    ---------
    >>> # Choose a random point on the GHZ surface
    >>> e1, e2, e3 = choose_GHZ_point()
    >>> print("Selected point (e1, e2, e3):", e1, e2, e3)
    """
    x = np.random.triangular(0, 0, 1)
    y = np.random.uniform(0, 1-x)
    e1 = 4*x*y + 2*y**2 -1
    e2 = 4*x**2*y + 12*x*y**2 - 8*x*y + 4*y**3 - 4*y**2 + 1
    e3 = -12*x**2*y - 12*x*y**2 + 12*x*y - 4*y**3 + 6*y**2 - 1
    if GHZ(e1, e2, e3, flip=True) >= 0:
        e1, e2, e3 = choose_GHZ_point()
    return e1, e2, e3


def choose_W1_point():
    """
    Randomly selects a point on the W green surface.

    The function generates random values for the model parameters `x` and `y`
    and computes the corresponding correlators `e1`, `e2`, and `e3`.

    Returns:
    --------
    tuple
        A tuple containing the correlators `(e1, e2, e3)` of a randomly selected point
        on the W green surface.

    Examples:
    ---------
    >>> # Choose a random point on the W green surface
    >>> e1, e2, e3 = choose_W1_point()
    >>> print("Selected point (e1, e2, e3):", e1, e2, e3)
    """
    x = np.linspace(0,0.19158750185069973, 1000)
    p = (-1 + 2*x + np.sqrt(5 - 8*x + 4*x**2))/2 - (1+x)/4 \
        - 1/4*np.sqrt((1+5*x**2-2*x**3)/(1-2*x))
    p = p/p.sum()
    x = np.random.choice(x, p=p)
    y = np.random.uniform((1+x)/4 + 1/4*np.sqrt((1+5*x**2-2*x**3)/(1-2*x)),
                          (-1 + 2*x + np.sqrt(5 - 8*x + 4*x**2))/2)
    e1 = -4*x*y + 2*x + 2*y - 1
    e2 = -8*x*y**2 + 8*x*y - 4*x + 4*y**2 - 4*y + 1
    e3 = 16*x**2*y - 8*x**2 - 8*x*y**2 - 4*x*y + 6*x + 4*y**2 - 2*y - 1
    return e1, e2, e3


def choose_W2_point():
    """
    Randomly selects a point on the W purple surface.

    The function generates random values for the model parameters `x` and `y`
    and computes the corresponding correlators `e1`, `e2`, and `e3`.

    Returns:
    --------
    tuple
        A tuple containing the correlators `(e1, e2, e3)` of a randomly selected point
        on the W purple surface.

    Examples:
    ---------
    >>> # Choose a random point on the W purple surface
    >>> e1, e2, e3 = choose_W2_point()
    >>> print("Selected point (e1, e2, e3):", e1, e2, e3)
    """
    x = np.linspace(1/2, 1, 1000)
    ymin = (x < 0.642129856069591) * (3*x**4 - 3*x**3 + 2*x - 1) / (x**4 + x**2 - x)
    p = 1 - ymin
    p = p / p.sum()
    x = np.random.choice(x, p=p)
    ymin = (x < 0.642129856069591) * (3*x**4 - 3*x**3 + 2*x - 1) / (x**4 + x**2 - x)
    y = np.random.uniform(ymin, 1)
    e1 = 1 - 2*x**2
    e2 = (-8*x**6 + 8*x**5*y + 16*x**5 - 12*x**4*y - 12*x**4 + 5*x**3*y + x**2*y + 6*x**2 - x*y - 4*x + 1) \
         / (x**3*y - 4*x**3 + x**2*y + 6*x**2 - x*y - 4*x + 1)
    e3 = (-8*x**6*y + 30*x**5*y - 38*x**4*y + 15*x**3*y + x**2*y + 4*x**2 - x*y - 4*x + 1) \
         / (x**3*y - 4*x**3 + x**2*y + 6*x**2 - x*y - 4*x + 1)
    purple_yellow_intersection = 16384*e1**27 - 180224*e1**26 + 1196032*e1**25 - 5742592*e1**24 + 20615168*e1**23 - 56135680*e1**22 + 120612352*e1**21 - 228131328*e1**20 + 458320896*e1**19 - 1046167168*e1**18 + 2331733888*e1**17 - 4407144448*e1**16 + 6688824388*e1**15 - 8023299660*e1**14 + 7537931916*e1**13 - 5464621696*e1**12 + 2968389493*e1**11 - 1138141837*e1**10 + 265563335*e1**9 - 16700987*e1**8 - 8983562*e1**7 + 1998514*e1**6 + 100954*e1**5 - 49974*e1**4 - 2847*e1**3 + 663*e1**2 + 67*e1 + e2**10*(6561*e1**11 + 104247*e1**10 + 219915*e1**9 - 622035*e1**8 - 492950*e1**7 + 1268998*e1**6 + 149366*e1**5 - 1055750*e1**4 + 279685*e1**3 + 283955*e1**2 - 161553*e1 + 21609) + e2**9*(-10692*e1**13 - 90504*e1**12 - 414306*e1**11 - 1303862*e1**10 + 342246*e1**9 + 9764966*e1**8 - 4931124*e1**7 - 21622172*e1**6 + 26988296*e1**5 - 897276*e1**4 - 17563274*e1**3 + 13568626*e1**2 - 4421386*e1 + 569982) + e2**8*(4356*e1**15 + 21812*e1**14 + 987308*e1**13 + 1396672*e1**12 - 6210143*e1**11 + 21651047*e1**10 - 22295541*e1**9 - 87149631*e1**8 + 221289198*e1**7 - 107942342*e1**6 - 213655678*e1**5 + 386501762*e1**4 - 289526851*e1**3 + 118724443*e1**2 - 26090889*e1 + 2386637) + e2**7*(-59776*e1**16 - 70240*e1**15 - 1917472*e1**14 - 9380048*e1**13 + 28379424*e1**12 + 16867080*e1**11 - 279458408*e1**10 + 734910248*e1**9 - 629758968*e1**8 - 845892752*e1**7 + 2885394128*e1**6 - 3642971008*e1**5 + 2679690032*e1**4 - 1232752024*e1**3 + 347338040*e1**2 - 53998936*e1 + 3434920) + e2**6*(42368*e1**18 - 258176*e1**17 + 3621632*e1**16 - 6251728*e1**15 + 1289008*e1**14 + 136980016*e1**13 - 714921024*e1**12 + 1772299586*e1**11 - 2085218770*e1**10 - 941497034*e1**9 + 8238330954*e1**8 - 15755567356*e1**7 + 17613546044*e1**6 - 12982248484*e1**5 + 6454923540*e1**4 - 2122224982*e1**3 + 433777350*e1**2 - 48347362*e1 + 2154498) + e2**5*(-276992*e1**19 + 2075904*e1**18 - 14173440*e1**17 + 20513664*e1**16 + 103670752*e1**15 - 770593760*e1**14 + 2463469256*e1**13 - 4359851312*e1**12 + 2726655316*e1**11 + 7436492636*e1**10 - 25857891836*e1**9 + 42706972388*e1**8 - 46086717208*e1**7 + 34696298936*e1**6 - 18411541040*e1**5 + 6763190808*e1**4 - 1643210460*e1**3 + 242371468*e1**2 - 18449116*e1 + 477940) + e2**4*(239104*e1**21 - 1102336*e1**20 + 1242112*e1**19 + 16131712*e1**18 - 43681664*e1**17 - 126848512*e1**16 + 1168480472*e1**15 - 3840545352*e1**14 + 7044753704*e1**13 - 5756144448*e1**12 - 6652288886*e1**11 + 30264043526*e1**10 - 53346579954*e1**9 + 60610400826*e1**8 - 48436723844*e1**7 + 27701832564*e1**6 - 11191339196*e1**5 + 3071159604*e1**4 - 530371262*e1**3 + 49258798*e1**2 - 1432346*e1 - 54542) + e2**3*(-729088*e1**22 + 4634624*e1**21 - 11909120*e1**20 - 7633920*e1**19 + 107279872*e1**18 - 126346752*e1**17 - 867555968*e1**16 + 4710416480*e1**15 - 12377991392*e1**14 + 20742551280*e1**13 - 22268054240*e1**12 + 11376413384*e1**11 + 7725071448*e1**10 - 22773516824*e1**9 + 25151063688*e1**8 - 17430231120*e1**7 + 8119680400*e1**6 - 2497965120*e1**5 + 464366128*e1**4 - 38724696*e1**3 - 1527688*e1**2 + 482024*e1 - 19160) + e2**2*(90112*e1**24 + 36864*e1**23 - 2674688*e1**22 + 10257408*e1**21 - 2399232*e1**20 - 67424256*e1**19 + 37204608*e1**18 + 1152374912*e1**17 - 6026605824*e1**16 + 17350357168*e1**15 - 34451561936*e1**14 + 50460275056*e1**13 - 55641835968*e1**12 + 45895091069*e1**11 - 27301467461*e1**10 + 10556212959*e1**9 - 1658238007*e1**8 - 686935502*e1**7 + 478329438*e1**6 - 103622770*e1**5 - 1074062*e1**4 + 4269809*e1**3 - 555801*e1**2 - 17837*e1 + 6101) + e2*(-81920*e1**25 + 147456*e1**24 + 2056192*e1**23 - 13893632*e1**22 + 42448896*e1**21 - 91543552*e1**20 + 267576832*e1**19 - 1138500352*e1**18 + 4113586944*e1**17 - 10947104640*e1**16 + 21771901984*e1**15 - 33147992864*e1**14 + 39132626140*e1**13 - 35836081032*e1**12 + 25130027838*e1**11 - 13072332822*e1**10 + 4699540998*e1**9 - 948145114*e1**8 - 16371668*e1**7 + 67599748*e1**6 - 16750872*e1**5 + 1069444*e1**4 + 242518*e1**3 - 48046*e1**2 + 598*e1 + 446) + 1
    if e1 < 1/3 or 4*e2 < 2*e1 - 1 or purple_yellow_intersection < 0:
        e1, e2, e3 = choose_W2_point()
    return e1, e2, e3


def choose_W3_point():
    """
    Randomly selects a point on the W red surface.

    The function generates random values for the model parameters `x` and `y`
    and computes the corresponding correlators `e1`, `e2`, and `e3`.

    Returns:
    --------
    tuple
        A tuple containing the correlators `(e1, e2, e3)` of a randomly selected point
        on the W red surface.

    Examples:
    ---------
    >>> # Choose a random point on the W red surface
    >>> e1, e2, e3 = choose_W3_point()
    >>> print("Selected point (e1, e2, e3):", e1, e2, e3)
    """
    y = np.linspace(1/3, 1, 1000)
    xmin = np.maximum(1 + 12*y + 3*y**2 - np.sqrt((1 + 12*y + 3*y**2)**2 - 192*y**2),
                      12*y*(2 + y - np.sqrt(2 + 2*y + y**2)))/(24*y)
    xmax = np.empty_like(y)
    for i, y_ in enumerate(y):
        roots = Polynomial([4 * y_ ** 2 * (1 + y_) ** 2,
                            -(5 + 16 * y_ + 50 * y_ ** 2 + 24 * y_ ** 3 + y_ ** 4) * y_,
                            1 + 12 * y_ + 34 * y_ ** 2 + 84 * y_ ** 3 + 45 * y_ ** 4,
                            -12 * y_ * (1 + 2 * y_ + 5 * y_ ** 2),
                            16 * y_ ** 2]).roots()
        xmax[i] = roots[~np.iscomplex(roots)].real.min()
    p = xmax - xmin
    p = p/p.sum()
    y = np.random.choice(y, p=p)
    xmin = np.maximum(1 + 12*y + 3*y**2 - np.sqrt((1 + 12*y + 3*y**2)**2 - 192*y**2),
                      12*y*(2 + y - np.sqrt(2 + 2*y + y**2)))/(24*y)
    xmax = Polynomial([4 * y_ ** 2 * (1 + y_) ** 2,
                       -(5 + 16 * y_ + 50 * y_ ** 2 + 24 * y_ ** 3 + y_ ** 4) * y_,
                       1 + 12 * y_ + 34 * y_ ** 2 + 84 * y_ ** 3 + 45 * y_ ** 4,
                       -12 * y_ * (1 + 2 * y_ + 5 * y_ ** 2),
                       16 * y_ ** 2]).roots().real.min()
    x = np.random.uniform(xmin, xmax)
    e1, e2, e3 = (-y * (3 * x - 1) / (x - y),
                  (4 * x ** 3 * y - 4 * x ** 2 * y ** 2 - 6 * x * y - x + 3 * y) / (x - y),
                  3 * y * (4 * x ** 3 - 4 * x ** 2 * y - 8 * x ** 2 + 2 * x * y + 5 * x - 1) / (x - y))
    red_cyan_intersection = 2048*e1**12 - 1024*e1**11*e2 - 3832*e1**11 + 2196*e1**10*e2 + 5508*e1**10 + 1062*e1**9*e2**2 + 2796*e1**9*e2 + 12198*e1**9 - 729*e1**8*e2**3 - 543*e1**8*e2**2 - 23427*e1**8*e2 - 20061*e1**8 - 1404*e1**7*e2**3 + 12636*e1**7*e2**2 + 29532*e1**7*e2 + 8084*e1**7 + 1368*e1**6*e2**3 - 2112*e1**6*e2**2 - 27680*e1**6*e2 - 1416*e1**6 - 2484*e1**5*e2**4 - 156*e1**5*e2**3 + 43568*e1**5*e2**2 + 2324*e1**5*e2 + 9212*e1**5 - 3456*e1**4*e2**4 - 18162*e1**4*e2**3 + 19706*e1**4*e2**2 - 25430*e1**4*e2 - 17250*e1**4 - 2664*e1**3*e2**4 - 17620*e1**3*e2**3 + 33876*e1**3*e2**2 + 44052*e1**3*e2 + 9340*e1**3 + 432*e1**2*e2**5 + 848*e1**2*e2**4 - 20128*e1**2*e2**3 - 31704*e1**2*e2**2 - 12700*e1**2*e2 - 500*e1**2 + 2268*e1*e2**4 + 5100*e1*e2**3 + 2682*e1*e2**2 - 624*e1*e2 - 506*e1 + 243*e2**3 + 477*e2**2 + 369*e2 + 71
    if red_cyan_intersection > 0:
        e1, e2, e3 = choose_W3_point()
    return e1, e2, e3


def choose_W4_point():
    """
    Randomly selects a point on the W yellow surface.

    The function generates random values for the model parameters `x` and `y`
    and computes the corresponding correlators `e1`, `e2`, and `e3`.

    Returns:
    --------
    tuple
        A tuple containing the correlators `(e1, e2, e3)` of a randomly selected point
        on the W yellow surface.

    Examples:
    ---------
    >>> # Choose a random point on the W yellow surface
    >>> e1, e2, e3 = choose_W4_point()
    >>> print("Selected point (e1, e2, e3):", e1, e2, e3)
    """
    ymax = Polynomial([4, -9, 11, -19, 18, -3, -13, 20, -10, -4, 4]).roots()[6].real
    y = np.linspace(0, ymax, 1000)
    xmin = np.empty_like(y)
    xmax = np.empty_like(y)

    def calculate_x_bounds(y):
        poly_min = Polynomial([-4*y**2,
                               1 + 12*y**2 - 4*y**3,
                               -2 - 10*y**2 + 4*y**3,
                               1 + 2*y + 4*y**3 - 3*y**4,
                               -2*y + 2*y**2 - 2*y**3 + 2*y**4,
                               y**2 - 2*y**3 + y**4])
        poly_max = Polynomial([1,
                               -3 - y + 3*y**2,
                               2 + 3*y - 8*y**2 + 2*y**3 + y**4,
                               -2*y + 4*y**2 - y**3 - 2*y**4 + y**5])
        roots_min = poly_min.roots()
        roots_max = poly_max.roots()
        xmin = roots_min[~np.iscomplex(roots_min)].real.min()
        xmax = roots_max[~np.iscomplex(roots_max)].real.min()
        return xmin, xmax

    for i, y_ in enumerate(y):
        xmin[i], xmax[i] = calculate_x_bounds(y_)
    p = xmax - xmin
    p = p/p.sum()
    y = np.random.choice(y, p=p)
    xmin, xmax = calculate_x_bounds(y)
    x = np.random.uniform(xmin, xmax)

    z = (np.sqrt(x) * (y - 1) * (x - y - 1) * np.sqrt(x ** 5 * y ** 4 - 2 * x ** 5 * y ** 3 + x ** 5 * y ** 2 + 2 * x ** 4 * y ** 4 - 2 * x ** 4 * y ** 3 + 2 * x ** 4 * y ** 2 - 2 * x ** 4 * y - 3 * x ** 3 * y ** 4 + 4 * x ** 3 * y ** 3 + 2 * x ** 3 * y + x ** 3 + 4 * x ** 2 * y ** 3 - 10 * x ** 2 * y ** 2 - 2 * x ** 2 - 4 * x * y ** 3 + 12 * x * y ** 2 + x - 4 * y ** 2) - x ** 4 * y ** 3 + 2 * x ** 4 * y ** 2 - x ** 4 * y + x ** 3 * y ** 4 - 2 * x ** 3 * y ** 3 + x ** 3 - x ** 2 * y ** 4 + 2 * x ** 2 * y ** 3 + 2 * x ** 2 * y ** 2 - x ** 2 * y - 2 * x ** 2 - 5 * x * y ** 2 + 4 * x * y + x + 2 * y ** 2 - 2 * y) / (2 * x ** 3 * y ** 5 - 2 * x ** 3 * y ** 4 - 4 * x ** 3 * y ** 3 + 6 * x ** 3 * y ** 2 - 2 * x ** 3 * y - 2 * x ** 2 * y ** 4 + 4 * x ** 2 * y ** 3 - 4 * x ** 2 * y + 2 * x ** 2 + 2 * x * y ** 4 - 2 * x * y ** 3 - 6 * x * y ** 2 + 10 * x * y - 4 * x + 2 * y ** 2 - 4 * y + 2)
    e1 = (-2 * x ** 2 * y ** 2 * z + 2 * x ** 2 * y * z + 2 * x * y * z - 2 * x * y - 2 * x * z + x - 2 * y * z + y + 2 * z - 1) / (x - y - 1)
    e2 = (4 * x ** 4 * y ** 3 * z ** 2 - 8 * x ** 4 * y ** 2 * z ** 2 + 4 * x ** 4 * y * z ** 2 + 4 * x ** 3 * y ** 5 * z ** 2 - 8 * x ** 3 * y ** 4 * z ** 2 - 4 * x ** 3 * y ** 3 * z ** 2 + 4 * x ** 3 * y ** 3 * z + 12 * x ** 3 * y ** 2 * z ** 2 - 8 * x ** 3 * y * z + x ** 3 * y - 4 * x ** 3 * z ** 2 + 4 * x ** 3 * z - x ** 3 - 4 * x ** 2 * y ** 4 * z ** 2 + 4 * x ** 2 * y ** 4 * z + 12 * x ** 2 * y ** 3 * z ** 2 - 12 * x ** 2 * y ** 3 * z - 4 * x ** 2 * y ** 2 * z + 2 * x ** 2 * y ** 2 - 20 * x ** 2 * y * z ** 2 + 24 * x ** 2 * y * z - 4 * x ** 2 * y + 12 * x ** 2 * z ** 2 - 12 * x ** 2 * z + 3 * x ** 2 + 4 * x * y ** 4 * z ** 2 - 8 * x * y ** 3 * z ** 2 + 4 * x * y ** 3 * z + x * y ** 3 - 8 * x * y ** 2 * z ** 2 + 8 * x * y ** 2 * z - 3 * x * y ** 2 + 24 * x * y * z ** 2 - 24 * x * y * z + 5 * x * y - 12 * x * z ** 2 + 12 * x * z - 3 * x + 4 * y ** 2 * z ** 2 - 4 * y ** 2 * z + y ** 2 - 8 * y * z ** 2 + 8 * y * z - 2 * y + 4 * z ** 2 - 4 * z + 1) / ((x - y - 1) ** 2 * (x * y - x + 1))
    e3 = (8 * x ** 4 * y ** 4 * z ** 2 - 12 * x ** 4 * y ** 3 * z ** 2 + 4 * x ** 4 * y * z ** 2 - 2 * x ** 3 * y ** 5 * z ** 2 - 2 * x ** 3 * y ** 4 * z ** 2 - 8 * x ** 3 * y ** 3 * z ** 2 + 14 * x ** 3 * y ** 3 * z + 26 * x ** 3 * y ** 2 * z ** 2 - 18 * x ** 3 * y ** 2 * z - 10 * x ** 3 * y * z ** 2 + x ** 3 * y - 4 * x ** 3 * z ** 2 + 4 * x ** 3 * z - x ** 3 + 2 * x ** 2 * y ** 4 * z ** 2 - 2 * x ** 2 * y ** 4 * z + 16 * x ** 2 * y ** 3 * z ** 2 - 8 * x ** 2 * y ** 3 * z - 24 * x ** 2 * y ** 2 * z ** 2 + 8 * x ** 2 * y ** 2 * z + 4 * x ** 2 * y ** 2 - 8 * x ** 2 * y * z ** 2 + 16 * x ** 2 * y * z - 6 * x ** 2 * y + 14 * x ** 2 * z ** 2 - 14 * x ** 2 * z + 3 * x ** 2 - 2 * x * y ** 4 * z ** 2 - 2 * x * y ** 3 * z ** 2 - 2 * x * y ** 3 * z + x * y ** 3 - 6 * x * y ** 2 * z ** 2 + 14 * x * y ** 2 * z - 5 * x * y ** 2 + 26 * x * y * z ** 2 - 28 * x * y * z + 9 * x * y - 16 * x * z ** 2 + 16 * x * z - 3 * x + 6 * y ** 2 * z ** 2 - 6 * y ** 2 * z + y ** 2 - 12 * y * z ** 2 + 12 * y * z - 4 * y + 6 * z ** 2 - 6 * z + 1) / ((x - y - 1) ** 2 * (x * y - x + 1))

    if e3 >= 0 or W1(e1, e2, e3) >= 0 or W2(e1, e2, e3) >= 0 or W3(e1, e2, e3) >= 0 or W5(e1, e2, e3) >= 0:
        e1, e2, e3 = choose_W4_point()

    return e1, e2, e3


def choose_W5_point():
    """
    Randomly selects a point on the W cyan surface.

    The function generates random values for the model parameters `x` and `y`
    and computes the corresponding correlators `e1`, `e2`, and `e3`.

    Returns:
    --------
    tuple
        A tuple containing the correlators `(e1, e2, e3)` of a randomly selected point
        on the W cyan surface.

    Examples:
    ---------
    >>> # Choose a random point on the W cyan surface
    >>> e1, e2, e3 = choose_W5_point()
    >>> print("Selected point (e1, e2, e3):", e1, e2, e3)
    """
    e1 = np.random.uniform(np.sqrt(5) - 2, 1/2)
    e2 = np.random.uniform(-5/27, 9 - 4*np.sqrt(5))
    e3 = W5_point(e1, e2)[0]

    if np.isinf(e3):
        e1, e2, e3 = choose_W5_point()

    return e1, e2, e3
