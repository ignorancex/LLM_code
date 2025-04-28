from diffsph.utils.consts import *

from scipy import special as sp

def Fav(x):
    """
    Synchrotron-power function for randomly-oriented magnetic fields [*]_. 

    .. math::
        F(x) = x^2 \\left(K_{4/3}(x) K_{1/3}(x) - \\frac35 x [K_{4/3}^2(x) - K_{1/3}^2(x)]\\right)

    :return: Pitch-angle averaged synchrotron function as a function of :math:`x`
    
    .. [*] Formula extracted from `Ghisellini et al, 1988`_ 
    
    .. _`Ghisellini et al, 1988`: https://ui.adsabs.harvard.edu/abs/1988ApJ...334L...5G%2F/
    """
    return x**2 * (sp.kv(4/3, x) * sp.kv(1/3, x) - 3/5 * x * (sp.kv(4/3, x)**2 - sp.kv(1/3, x)**2)) if x > 1e-115 else 1e-115**2 * (sp.kv(4/3, 1e-115) * sp.kv(1/3, 1e-115) - 3/5*1e-115 * (sp.kv(4/3, 1e-115)**2 - sp.kv(1/3, 1e-115)**2))*(x/1e-115)**(1/3)


def M_raw(xi, eta, delta):
    """
    \"Raw\" master function
    
    .. math::
        \\mathcal M(\\xi,\\eta,\\delta) = \\int_\\xi^\\infty dx F(x^2)\\exp\\left(-\\eta\\,[x^{1-\\delta}-\\xi^{1-\\delta}]\\right)
        
    :return: above integral
    """
    return quad(lambda x: Fav(x**2) * np.exp(- eta * (x**(1-delta)-xi**(1-delta))), xi, min(15, 100 / eta**(1/(1-delta))))[0]


def M_C(xi, eta, delta):
    '''
    Master function in the Regime-C limit
    
    .. math::
        \\mathcal M_C(\\xi,\\eta,\\delta) = \\frac{\\xi^\\delta}{(1-\\delta)\\eta} F(\\xi^2)
        
    '''
    return xi**delta / (1 - delta) / eta * Fav(xi**2)


def M_i(xi, eta, delta):
    '''
    Master function in the large :math:`\\eta` limit
    
    .. math::
        \\mathcal M_i(\\xi,\\eta,\\delta) = \\frac{\\Gamma^2(1/3)\\eta^{-\\frac{5}{3(1-\\delta)}}}{5\sqrt[3]{2}(1-\\delta)}\\Gamma\\left(\\frac{5}{3(1-\\delta)},\\eta\\, \\xi^{1 - \\delta}\\right)\\exp\\left(\\eta\\,  \\xi^{1-\\delta}\\right)

    '''
    return sp.gamma(1/3)**2 / 5 / 2**(1/3) / (1 - delta) * sp.gamma(5 / 3 / (1 - delta)) *\
    sp.gammaincc(
        5 / 3 / (1 - delta),
        eta * xi ** (1 - delta)
    ) * eta ** (- 5 / 3 / (1 - delta)) * np.exp(eta * xi**(1 - delta))


def anltc_Mst(xi, eta, delta):
    """
    Master function
    
    .. math::
        \\mathcal M(\\xi,\\eta,\\delta) = \\int_\\xi^\\infty dx F(x^2)\\exp\\left(-\\eta\\,[x^{1-\\delta}-\\xi^{1-\\delta}]\\right)
        
    .. note:: Function evaluates the above integral only for those values where no numerical errors are present. Otherwise, it uses the approximate formulas :py:func:`diffsph.spectra.analytics.M_C` or :py:func:`diffsph.spectra.analytics.M_i` 
    """
    if eta > 25 * xi**( - (1 - delta) ):
        return M_C(xi, eta, delta)
    elif 10 ** 4 < eta <= 25 * xi **( - (1 - delta) ):
        return M_i(xi, eta, delta)
    else:
        return M_raw(xi, eta, delta)


# # Additional stuff

def btot(E, B):
    """
    Total energy loss function in GeV/s

    :param E: cosmic-ray energy in GeV
    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G

    :return: energy-loss rate in GeV/s
    """
    return b0 * (1 + (B / Bc)**2) * E**2


def lam(E, B, D0, delta=1/3):
    """
    Syrovatskii variable in kpc\ :sup:`2`

    :param E: cosmic-ray energy in GeV
    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm\ :sup:`2`/s
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy (default value = 1/3) 

    :return: Syrovatskii variable in kpc\ :sup:`2`
    """
    return D0 / b0 / (1 + (B / Bc)**2) / (1 - delta) / E**(1 - delta) * cm_to_kpc**2
