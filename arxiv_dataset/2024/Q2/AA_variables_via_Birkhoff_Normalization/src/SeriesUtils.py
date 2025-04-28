import numpy as np
from scipy.special import binom, gamma
factorial = lambda n: gamma(n+1)
from celmech.poisson_series import PSTerm, PoissonSeries
from collections import defaultdict
import sympy as sp

def PotentialDerivativesToPoissonSeries(derivs_dict,nmax,cvar_symbols=None):
    r"""
    Convert a dictionary containing mixed partial derivatives up to order `nmax`
    of a 2D effective potential expressed in polar coordinates into a collection
    of Poisson series in complex canonical variables with terms grouped by
    order. 

    Parameters
    ----------
    derivs_dict : dict
        Dictionary of partial derivatives with the key (l,m) denoting .. math::
            \frac{\partial^{l+m}} {\partial R^{l}\partial z^m}\Phi_\mathrm{eff}

    nmax : int
        maximum order of derivatives appearing in `derivs_dict`
    
    cvar_symbols : tuple, optional
        Symbols to use for complex variables. Default behavior of celmech will be 
        used in case they are note specified.
    
    Returns
    -------
    dict
        A dictionary with integer keys and PoissonSeries values containing terms
        of a given order, specified by the key.
    """
    kappa_sq = derivs_dict[(2,0)]
    nu_sq = derivs_dict[(0,2)]
    kappa = np.sqrt(kappa_sq)
    nu = np.sqrt(nu_sq)
    dR_factor = np.sqrt(0.5 / kappa)
    dz_factor = np.sqrt(0.5 / nu)
    Hdict = dict()
    Hdict[2] = PoissonSeries.from_PSTerms(
        [
            PSTerm(kappa,[1,0],[1,0],[],[]),
            PSTerm(nu,[0,1],[0,1],[],[])
        ],
        cvar_symbols = cvar_symbols
    )
    for n in range(3,nmax+1):
        term_list = []
        for m in range(0,n+1):
            if (n-m)%2:
                continue
            prefactor_nm = 1/factorial(m)/factorial(n-m)
            prefactor_nm *= derivs_dict[(m,n-m)]
            prefactor_nm *= dR_factor**m
            prefactor_nm *= dz_factor**(n-m)
            for p in range(m+1):
                binom_p = binom(m,p)
                for q in range(0,n-m+1):
                    binom_q = binom(n-m,q)
                    C = float(binom_p*binom_q*prefactor_nm)
                    term = PSTerm(C,[p,q],[m-p,n-m-q],[],[])
                    term_list.append(term)
        Hdict[n] = PoissonSeries.from_PSTerms(term_list,cvar_symbols = cvar_symbols)
    return Hdict

def get_xseries(N):
    """
    Get a list series dictionaries representing the complex canonical variables
    :math:`x_i`.

    Parameters
    ----------
    N : int
        Number of complex canonical variables

    Returns
    -------
    list
        list of `defaultdict` objects
    """
    xseries_list = []
    for i in range(N):
        zeros = np.zeros(N,dtype = int)
        oi = zeros.copy()
        oi[i] = 1
        xseries = defaultdict(lambda: PoissonSeries(N,0))
        xseries[1] = PoissonSeries.from_PSTerms([PSTerm(1,oi,zeros,[],[])])
        xseries_list.append(xseries)
    return xseries_list

def pert_series_to_PoissonSeries(pert_series,lmax = None):
    """
    Convert a dictionary grouping terms by order in a perturbation parameter to
    into a `celmech.poisson_series.PoissonSeries` object

    Parameters
    ----------
    pert_series : dict
        Dictionary containing perturbative expansion
    lmax : int, optional
        maximum order of terms to include in expansion. By default the highest
        order terms appearing in the dictionary will be included.

    Returns
    -------
    celmech.poisson_series.PoissonSeries
        Resulting series.
    """
    if not lmax:
        lmax = np.inf
    full_series = PoissonSeries(2,0)
    for order,series in pert_series.items():
        if order <= lmax:
            full_series += series
    return full_series

def collect_by_rt_action_power(series,indx):
    rtJ_coeffs = defaultdict(list)
    for term in series.terms:
        powr = term.k[indx] + term.kbar[indx]
        rtJ_coeffs[powr].append(term)
    return {pwr:PoissonSeries.from_PSTerms(term_list) for pwr,term_list in rtJ_coeffs.items()}

def h_series_to_omega_series(H_series):
    r"""
    Construct series for the dynamical frequencies :math:`\Omega_i =
    \partial_{J_i}H(J)` given an inpu Hamiltonian.

    Parameters
    ----------
    H_series : celmech.poisson_series.PoissonSeries
        Series representing the Hamiltonian. 

    Returns
    -------
    list
        A list of celmech.poisson_series.PoissonSeries objects that give
        frequencies as a function of complex variables.
    """
    N = H_series.N
    Omega_term_lists = [[] for i in range(N)]
    oi = np.eye(N,dtype=int)
    for term in H_series.terms:
        C,k,kbar = term.C,term.k,term.kbar
        for i in range(N):
            if k[i]>0:
                Oterm = PSTerm(k[i] * C,kbar - oi[i],kbar - oi[i],[],[])
                Omega_term_lists[i].append(Oterm)
    return [PoissonSeries.from_PSTerms(term_list) for term_list in Omega_term_lists]

def ic_to_xs(ic,Rc,omega):
    R,z,pR,pz = ic
    kappa,nu = omega
    xR = np.sqrt(0.5 * kappa) * ((R-Rc) + 1j * pR/kappa)
    xz = np.sqrt(0.5 * nu) * (z + 1j * pz/nu)
    return xR,xz

def to_pade1_approx_function(series,indx = 1):
    r"""
    Given a `PoissonSeries` object, return a function that evaluates a (1,n)
    Padé approximant, where the Padé approximant is developed in the action
    variable associated with the complex variables occuring in the position
    specified by `indx`.

    Parameters
    ----------
    series : celmech.poisson_series.PoissonSeries
        Poisson series from which Padé approximant should be built.
    indx : int, optional
        specify which variable to establish Padé approximant in, by default 1.

    Returns
    -------
    function
        A function that takes as arguemnts complex canonical variables (except
        those that are being Padé-approximated) followed by
        :math:`J=x_i\bar{x}_i` and :math:`x_i/|x_i|` where i is specified by the
        arguemnt `indx` 
    """

    N = series.N
    cvars = series.cvar_symbols
    J = sp.symbols("J",positive = True)
    Z = sp.symbols("Z")
    rule = {cvars[indx]:Z,cvars[indx+N]:1/Z}
    to_exprn = lambda s: s.as_expression().xreplace(rule)
    series_by_rt_action = collect_by_rt_action_power(series,indx)
    lmax =  max(list(series_by_rt_action.keys()))
    # get pade coefficients
    b = -1  * to_exprn(series_by_rt_action[lmax])/to_exprn(series_by_rt_action[lmax-2])
    ci_minus_1 = 0
    numerator = 0
    for i in range(lmax//2):
        ci = to_exprn(series_by_rt_action.get(2*i,PoissonSeries(N,0)))
        ai = ci + b * ci_minus_1
        numerator += ai * J**i
        ci_minus_1 = ci
    exprn = numerator/(1 + b * J)
    args = [element for index, element in enumerate(cvars) if index not in (indx,indx+N)]
    args+= [J,Z]
    pade_fn = sp.lambdify(args,exprn)
    return pade_fn

def to_pade2_approx_function(series,indx = 1):
    r"""
    Given a `PoissonSeries` object, return a function that evaluates a (2,n)
    Padé approximant, where the Padé approximant is developed in the action
    variable associated with the complex variables occuring in the position
    specified by `indx`.

    Parameters
    ----------
    series : celmech.poisson_series.PoissonSeries
        Poisson series from which Padé approximant should be built.
    indx : int, optional
        specify which variable to establish Padé approximant in, by default 1.

    Returns
    -------
    function
        A function that takes as arguemnts complex canonical variables (except
        those that are being Padé-approximated) followed by
        :math:`J=x_i\bar{x}_i` and :math:`x_i/|x_i|` where i is specified by the
        arguemnt `indx` 
    """
    N = series.N
    cvars = series.cvar_symbols
    J = sp.symbols("J",positive = True)
    Z = sp.symbols("Z")
    rule = {cvars[indx]:Z,cvars[indx+N]:1/Z}
    to_exprn = lambda s: s.as_expression().xreplace(rule)
    series_by_rt_action = collect_by_rt_action_power(series,indx)
    lmax =  max(list(series_by_rt_action.keys()))
    # get pade coefficients
    c_n = to_exprn(series_by_rt_action[lmax])
    c_nm1 = to_exprn(series_by_rt_action[lmax-2])
    c_nm2 = to_exprn(series_by_rt_action[lmax-4])
    c_nm3 = to_exprn(series_by_rt_action[lmax-6])
    det = c_nm2 * c_nm2 - c_nm1 * c_nm3
    b1 = (   c_nm3 * c_n - c_nm2  * c_nm1) / det
    b2 = (-1*c_nm2 * c_n + c_nm1  * c_nm1) / det
    ci_minus_1 = 0
    ci_minus_2 = 0
    numerator = 0
    for i in range(lmax//2-1):
        ci = to_exprn(series_by_rt_action.get(2*i,PoissonSeries(N,0)))
        ai = ci + b1 * ci_minus_1 + b2 * ci_minus_2
        numerator += ai * J**i
        ci_minus_2 = ci_minus_1
        ci_minus_1 = ci
    exprn = numerator/(1 + b1 * J + b2 * J*J)
    args = [element for index, element in enumerate(cvars) if index not in (indx,indx+N)]
    args+= [J,Z]
    pade_fn = sp.lambdify(args,exprn)
    return pade_fn

def generate_phidot_series(Rc,L,kappa,lmax):
    r"""
    Generate perturbative expansion for :math:`\frac{d}{dt}\phi`.

    Parameters
    ----------
    Rc : float
        Circular orbit radius
    L : float
        Angular momentum
    kappa : float
        epicyclic frequency
    lmax : int
        maximum order

    Returns
    -------
    defaultdict
        Dictionary containing term in the expansion of :math:`L/R^2`
    """
    phidot_series = defaultdict(lambda: PoissonSeries(2,0))
    Omega_phi0 = L / Rc / Rc
    phidot_series[0] = PSTerm(Omega_phi0,[0,0],[0,0],[],[]).as_series()
    o1 = np.array([1,0],dtype=int)
    for l in range(1,lmax+1):
        prefactor = Omega_phi0 * (np.sqrt(1/2/kappa)/Rc)**l * (-1)**(l) * binom(1+l,l)
        terms = [PSTerm(prefactor * binom(l,m),m*o1,(l-m)*o1,[],[]) for m in range(l+1)]
        phidot_series[l] = PoissonSeries.from_PSTerms(terms)
    return phidot_series
