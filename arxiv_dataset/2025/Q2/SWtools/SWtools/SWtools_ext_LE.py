"""SWtools_ext_LE_v2.py

SWtools extension module implementing functions for the linear stability
analysis of generalized nonlinear Schrödinger equations.  The analysis proceeds
by calculating the linearized eigenspectrum (LE), describing small-amplitude
perturbations atop a solitary wave.

Note
----
This module implements two methods for solving for the LE of genarlized
nonlinear Schrödinger equations with different restictions:

LE_GNSE: this function is tailored towards a GNSE with with generic nonlinear
functional and linear differential operator of second order in the transverse
coordinate.

LE_HONSE: this function is tailored towards a GNSE with cubic nonlinear
functional and linear differential operator including dispersion of orders two,
three and four.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from scipy.sparse import diags
from scipy.linalg import eig

# -- CONVENIENT ABBREVIATIONS
FTFREQ = np.fft.fftfreq


def LE_GNSE(xi, U, kap, c2, F_fun, F1_fun):
    """Linearized eigenspectrum for the GNSE.

    Solves the eigenvalue problem for the linear stability matrix of a GNSE
    with generic nonlinear functional and linear differential operator of
    second order in the transverse coordinate [P1998,K1998].

    Note
    ----
    Assumes that the soliton solution is given by an even, positive,
    single-humped function U, see [P1998].

    References
    ----------
    [P1998] D. E. Pelinovsky, Y. S. Kivshar, V. V. Afanasjev, Internal modes of
    envelope solitons, Physica D 116 (1) (1998) 121–142,
    https://doi.org/10.1016/S0167-2789(98)80010-9.

    [K1998] Y. S. Kivshar, D. E. Pelinovsky, T. Cretegny, M. Peyrard, Internal
    modes of solitary waves, Phys. Rev. Lett. 80 (1998) 5032,
    https://doi.org/10.1103/PhysRevLett.80.5032.

    Parameters
    ----------
    xi : array_like
        Disrete transverse coordinates.
    U : array_like
        Solitary wave solution.
    kap : float
        Solitary wave wavenumber.
    c2 : float
        Constant coefficient of the linear differential operator part
    F_fun : function
        Nonlinear functional.
    F1_fun : function
        Derivative of the nonlinear functional.
    """

    # -- COMPOSE MATRIX DETERMINING EVP FOR THE PERTURBATION MODES
    # ... PREPARE COEFFICIENTS FOR FINITE-DIFFERENCE MATRIX 
    dxi = xi[1]-xi[0]
    dxi2= dxi**2
    I = np.abs(U)**2

    # ... SET UP DIAGONALS 
    D0 = -(30/12)*c2/dxi2*np.ones(xi.size)
    D1 =  (16/12)*c2/dxi2*np.ones(xi.size-1)
    D2 = -( 1/12)*c2/dxi2*np.ones(xi.size-2)

    # ... SET UP SPARSE FINITE DIFFERENCE MATRICES
    Z0 = diags([np.zeros_like(U)], [0]).toarray()
    L0 = diags([D0 + kap - F_fun(I), D1, D2, D1, D2], [0,-1,-2,1,2]).toarray()
    L1 = diags([D0 + kap - F_fun(I) - 2*I*F1_fun(I), D1, D2, D1, D2], [0,-1,-2,1,2]).toarray()

    # ... SET UP AUXILIARY MATRIX
    M = np.block([
      [ Z0, L0],
      [ L1, Z0]
    ])

    # ... SOLVE EIGENVALUE PROBLEM
    e_, v_ = eig(M)

    # -- FILTER FOR EIGENVALUES WITHIN GAP OF CONTINUOUS SPECTRUM
    # ... DETERMINE ONSET OF CONTINUOUS-WAVE BANDS
    Lam_max = kap
    # ... FILTER EIGENVALUES AND EIGENFUNCTIONS
    Lam_ = 1j*e_
    Lam = Lam_[ np.abs(np.imag(Lam_))<kap]
    v = v_[:, np.abs(np.imag(Lam_))<kap]
    # ... SEPARATE THE MODES AND MAKE INDEXING MORE INTUITIVE
    f = np.asarray([v[:xi.size,n] for n in range(Lam.size)])
    g = np.asarray([v[xi.size:,n] for n in range(Lam.size)])
    # ... GET ORDER FOR INCREASING MAGNITUDE OF IMAGINARY PART
    idx_srt = np.flip(np.argsort(np.abs(np.imag(Lam))))

    return kap, Lam[idx_srt], f[idx_srt], g[idx_srt]


def LE_HONSE(xi, U, kap, coeffs):
    """Linearized eigenspectrum for a HONSE.

    Solves the eigenvalue problem for the linear stability matrix of a GNSE
    with cubic nonlinear functional and linear differential operator including
    dispersion of orders two, three and four [T2020,M2024].

    Note
    ----
    The linear stability matrix for this model is generally non-selfadjoint and
    does not simply reduce to that of [P1998] (with higher orders of the linear
    derivatives included).

    References
    ----------
    [T2020] K. K. K. Tam, T. J. Alexander, A. Blanco-Redondo, C. M. de Sterke,
    Generalized dispersion Kerr solitons, Phys. Rev. A 101 (2020) 043822,
    https://doi.org/10.1103/PhysRevA.101.043822.

    [M2024] O. Melchert, A. Demircan,
    https://doi.org/10.48550/arXiv.2504.10623.

    [P1998] D. E. Pelinovsky, Y. S. Kivshar, V. V. Afanasjev, Internal modes of
    envelope solitons, Physica D 116 (1) (1998) 121–142,
    https://doi.org/10.1016/S0167-2789(98)80010-9.

    Parameters
    ----------
    xi : array_like
        Disrete transverse coordinates.
    U : array_like
        Solitary wave solution.
    kap : float
        Solitary wave wavenumber.
    coeffs : array_like
        Scaled dispersion parameters coeffs = [c0,c1,c2,c3,c4] defining the
        propagation constant (c_n = beta_n/n!).
    """

    # -- COMPOSE MATRIX DETERMINING EVP FOR THE PERTURBATION MODES
    # ... PREPARE COEFFICIENTS FOR FINITE-DIFFERENCE MATRIX 
    dxi = xi[1]-xi[0]
    c2, c3, c4 = coeffs
    sc2, sc3, sc4 = c2/dxi**2, c3/dxi**3, c4/dxi**4
    I = np.abs(U)**2

    # ... SET UP DIAGONALS 
    D3m = (               1j*( 1/8)*sc3 - ( 1/6)*sc4)*np.ones(xi.size-3)
    D2m = ( ( 1/12)*sc2 - 1j*sc3        +      2*sc4)*np.ones(xi.size-2)
    D1m = (-(16/12)*sc2 + 1j*(13/8)*sc3 - (13/2)*sc4)*np.ones(xi.size-1)
    D0  = ( (30/12)*sc2                 + (28/3)*sc4)*np.ones(xi.size)
    D1p = (-(16/12)*sc2 - (13/8)*1j*sc3 - (13/2)*sc4)*np.ones(xi.size-1)
    D2p = ( ( 1/12)*sc2 + 1j*sc3        +      2*sc4)*np.ones(xi.size-2)
    D3p = (              -1j*( 1/8)*sc3 - ( 1/6)*sc4)*np.ones(xi.size-3)

    # ... SET UP CORRESPONDING SPARSE FINITE DIFFERENCE MATRICES
    M11 = diags([D0 + (2*np.abs(U)**2-kap), D1p, D2p, D3p, D1m, D2m, D3m], [0,-1,-2,-3,1,2,3]).toarray()
    M12 = diags([U*U],[0]).toarray()

    # ... SET UP AUXILIARY MATRIX
    M = np.block([
      [ M11,  M12],
      [-np.conj(M12), -np.conj(M11)]
    ])

    # ... SOLVE EIGENVALUE PROBLEM
    e_, v_ = eig(M)

    # -- FILTER FOR EIGENVALUES WITHIN GAP OF CONTINUOUS SPECTRUM
    # ... DETERMINE ONSET OF CONTINUOUS-WAVE BANDS
    w = FTFREQ(xi.size, d=xi[1]-xi[0])*2*np.pi
    bw  = c2*w**2 + c3*w**3 + c4*w**4
    bw_max = np.max(bw)
    Lam_max = kap - bw_max
    # ... FILTER EIGENVALUES AND EIGENFUNCTIONS
    Lam_ = 1j*e_
    Lam = Lam_[ np.abs(np.imag(Lam_))<Lam_max]
    v = v_[:, np.abs(np.imag(Lam_))<Lam_max]
    # ... SEPARATE THE MODES AND MAKE INDEXING MORE INTUITIVE
    f = np.asarray([v[:xi.size,n] for n in range(Lam.size)])
    g = np.asarray([v[xi.size:,n] for n in range(Lam.size)])
    # ... GET ORDER FOR INCREASING MAGNITUDE OF IMAGINARY PART
    idx_srt = np.flip(np.argsort(np.abs(np.imag(Lam))))

    return Lam_max, Lam[idx_srt], f[idx_srt], g[idx_srt]


def LE_dump(Lam_max, Lam, f, g):
    """List discrete eigenvalues.

    Parameters
    ----------
    Lam_max : float
        Continuum edge.
    Lam : array_like
        Complex-valued eigenvalues.
    f, g : array_like
        Complex-valued eigenfunctions.
    """
    print("# -- DISCRETE EIGENSPECTRUM (SORTED)")
    print("# -- CONTINUUM EDGE:", Lam_max)
    for n in range(Lam.size):
        LamR, LamI = np.real(Lam[n]), np.imag(Lam[n])
        print(f"n = {n}  Re[Lam_n] = {LamR:+6.5F}  Im[Lam_n] = {LamI:+6.5F}")


