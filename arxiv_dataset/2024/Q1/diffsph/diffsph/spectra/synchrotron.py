from diffsph.utils.consts import *
from diffsph.utils.tools import *
from diffsph.spectra.analytics import *


from scipy.integrate import quad
from scipy.interpolate import interpn


# ########################################################################################################
# ########################################################################################################

cwdir = os.path.dirname(os.path.realpath(__file__))
alldata = load_data(os.path.join(cwdir,'Interpolations'))


# %%%%%%%%%%%%%%%%%%%
# Spectral function %
# %%%%%%%%%%%%%%%%%%%

# "Kernel" synch. spectrum

def htX(E, nu, tau, delta, B, fast_comp = True):
    """
    Spectral function kernel in erg/GHz :math:`\\hat X`

    :param E: CRE energy in GeV
    :param nu: frequency in GHz
    :param tau: diffusion time-scale parameter for a 1 GeV CRE in s
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy
    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
    :param bool fast_comp: if ``'True'``, employs the interpolating method (default value = ``'True'``)
        
    :return: spectral kernel in erg/GHz
    """
    if fast_comp:        
        return X0 * B / (1 + (B / Bc)**2) / Enu(B, nu) * Mst(
            Enu(B, nu) / E, eta(Enu(B, nu), B, tau, delta_float(delta)), delta
        )
    else:
        return X0 * B / (1 + (B / Bc)**2) / Enu(B, nu) * anltc_Mst(
            Enu(B, nu) / E, eta(Enu(B, nu), B, tau, delta_float(delta)), delta_float(delta))


# Spectral function for generic S(E) and synch. rad.

def X_gen(Emin, Emax, S_func, nu, tau, delta, B):
    """
    Spectral function in erg/GHz for generic CRE sources
    
    .. math::
        X_\\text{gen}(\\nu) = \\int_{E_m}^{E_M}dE'\\hat X(\\nu, E')S(E')

    :param Emin: low-E cutoff energy in GeV of the CRE source ``'S_func'``
    :param Emax: high-E cutoff energy in GeV of the CRE source ``'S_func'``
    :param S_func: CRE source function
    :param nu: frequency in GHz
    :param tau: diffusion time-scale parameter for a 1 GeV CRE in s
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy
    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
        
    :return: spectral function in erg/GHz
    """
    return quad(lambda En: S_func(En) * htX(nu, tau, B, delta, En), Emin, Emax)[0]
    

# # Model-specific (pre-computed) functions


def X(nu, tau, delta, B, hyp, **kwargs):
    """
    Spectral function in erg/GHz for all hypotheses built in diffsph 

    :param nu: frequency in GHz
    :param tau: diffusion time-scale parameter for a 1 GeV CRE in s
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy
    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
    :param str hyp: hypothesis: ``'wimp'``, ``'decay'`` or ``'generic'``
    
    |
    
    **Keyword arguments:**
    
    |
    
    - If ``hyp = 'wimp'`` or ``'decay'``
    
    :param mchi: mass of the DM particle in GeV/:math:`c^2`
    :param str channel: annihilation/decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.

    - If ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    
    |

    :return: spectral function in erg/GHz
    """
    if hyp.lower() in all_names['wimp']:
        return X_DM(k = 2, mchi = kwargs['mchi'], channel = kwargs['channel'], nu = nu, tau = tau, delta = delta, B = B)
    elif hyp.lower() in all_names['decay']:
        return X_DM(k = 1, mchi = kwargs['mchi'], channel = kwargs['channel'], nu = nu, tau = tau, delta = delta, B = B)
    elif hyp.lower() in all_names['generic']:
        return X_pw(Gamma = kwargs['Gamma'], nu = nu, tau = tau, delta = delta, B = B)
    else:
        print('Hypothesis not found')


########################################################################################################
########################################################################################################


# # # DM annihilation or decay 

def X_DM(k, mchi, channel, nu, tau, delta, B):
    """
    Spectral function in erg/GHz for all DM hypotheses built in diffsph

    :param k: hypothesis index (k=1 for decay and k=2 for annihilation)
    :param mchi: mass of the DM particle in GeV/:math:`c^2`
    :param channel: annihilation/decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    :param nu: frequency in GHz
    :param tau: diffusion time-scale parameter for a 1 GeV CRE in s
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy
    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
        
    :return: spectral function in erg/GHz
    """
    conds = [(channel == 'bb' and mchi <= 4 * 2 / k), 
             (channel == 'WW' and mchi <= 80 * 2 / k), 
             (channel == 'ZZ' and mchi <= 91 * 2 / k),
             (channel == 'hh' and mchi <= 125 * 2 / k),
             (channel == 'nunu' and mchi <= 125 * 2 / k),
             (channel == 'tt' and mchi <= 173 * 2 / k),
             mchi <= 2 / k
            ]
    if any(conds):
        return 0.
    
    return X0 * B / (1 + (B / Bc)**2) / Enu(B, nu) * Mst_DM(
        2 * Enu(B, nu) / k / mchi, eta(Enu(B, nu), B, tau, delta_float(delta)), k * mchi / 2, delta, channel
    )

# # Power-law CRE distribution 


def X_pw(Gamma, nu, tau, delta, B):
    """
    Spectral function in erg/GHz for the generic power-law hypothesis
        
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param nu: frequency in GHz
    :param tau: diffusion time-scale parameter for a 1 GeV CRE in s
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy
    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
        

    :return: spectral function in erg/GHz
    """
    return X0 * B / (1 + (B / Bc)**2) / Enu(B, nu) * (Enu(B, nu) / 1)**( - Gamma + 1) * Mst_pw(
        eta(Enu(B, nu), B, tau, delta_float(delta)), Gamma, delta
    ) / (Gamma - 1)


# ########################################################################################################
# ########################################################################################################


# # Interpolating functions

# # # Generic case

def lMst(Lxi, Leta, delta):
    """
    Interpolation function for (kernel) :math:`\\log(\\hat{\\mathcal M})` 
        
    :param Lxi: :math:`\\log(\\xi)`
    :param Leta: :math:`\\log(\\eta)`
    :param delta: :math:`\\delta`

    :return: :math:`\\log(\\hat{\\mathcal M})` as a function of :math:`\\log(\\xi)`, :math:`\\log(\\eta)` and :math:`\\delta`
    """
    #     Grid
    
    lxi = alldata['lxi']
    leta = alldata['leta']
    
    lpnts = (lxi, leta)

    #     Evaluations to be interpolated over 
    
    lmst = alldata[var_to_str(delta)+'m'].reshape(len(lxi), len(leta))
        
    return interpn(lpnts, lmst, np.array([Lxi, Leta]), bounds_error=False, fill_value=None)[0]


def Mst(xi, eta, delta):
    """
    Interpolation function for the kernel function :math:`\\hat{\\mathcal M}(\\xi,\\eta,\\delta)` 

    :param xi: :math:`\\xi`
    :param eta: :math:`\\eta`
    :param delta: :math:`\\delta`
        
    :return: Spectral-function kernel (as an interpolation function) 
    """
    return 10**(lMst(np.log10(xi), np.log10(eta), delta))



# # # DM annihilation or decay


def lMst_DM(Lxi, Leta, Lm, delta, channel):
    """
    Interpolation function :math:`\\log(\\mathcal M)` for DM hypotheses
    
    :param Lxi: :math:`\\log(\\xi)`
    :param Leta: :math:`\\log(\\eta)`
    :param Lm: :math:`\\log(m/\\text{GeV})` (:math:`m` is the WIMP mass)
    :param delta: :math:`\\delta`
    :param channel: annihilation/decay channel
    
    :return: :math:`\\log(\\mathcal M)` as a function of :math:`\\log(\\xi)`, :math:`\\log(\\eta)`, :math:`\\log(m)` and :math:`\\delta`
    """
    if channel == 'ee':
        return lMst(Lxi, Leta, delta)
    #     Grid
    
    lxi = alldata['lxi']
    leta = alldata['leta']
    lmss = alldata[ch_to_grid[channel]]

    lpnts = (lxi, leta, lmss)
    
    #     Evaluations to be interpolated over 
    
    lmst = alldata[var_to_str(delta)+channel].reshape(len(lxi), len(leta), len(lmss))
    
    return interpn(lpnts, lmst, np.array([Lxi, Leta, Lm]), bounds_error=False, fill_value=None)[0]


def Mst_DM(xi, eta, m, delta, channel):
    """
    Master function for dark-matter hypotheses

    :param xi: :math:`\\xi`
    :param eta: :math:`\\eta`
    :param delta: :math:`\\delta`
    :param m: WIMP mass in GeV
    :param channel: annihilation/decay channel
    
    :return: Master function (as an interpolation function) for DM hypotheses
    """
    if channel == 'ee':
        return Mst(xi, eta, delta)
    return 10 ** (lMst_DM(np.log10(xi), np.log10(eta), np.log10(m), delta, channel))


# # # Power law 

def lMst_pw(Leta, Gamma, delta):
    """
    Interpolation function :math:`\\log(\\mathcal M)` for the gereric power-law hypothesis

    :param Leta: :math:`\\log(\\eta)`
    :param Gamma: :math:`\\Gamma`
    :param delta: :math:`\\delta`

    :return: :math:`\\log(\\mathcal M_\\text{gen})` as a function of :math:`\\log(\\eta)`, :math:`\\Gamma` and :math:`\\delta`
    """
    #     Grid
 
    leta = alldata['letapw']
    gam = alldata['gam']
    
    lpnts = (leta, gam)
    
    #     Evaluations to be interpolated over 
    
    lmst = alldata[var_to_str(delta)+'pw'].reshape(len(leta), len(gam))
    
    return interpn(lpnts, lmst, np.array([Leta, Gamma]), bounds_error=False, fill_value=None)[0]


def Mst_pw(eta, Gamma, delta):
    """
    Master function for the generic power-law hypothesis
    
    :param eta: :math:`\\eta`
    :param Gamma: :math:`\\Gamma`
    :param delta: :math:`\\delta`

    :return: Master function (as an interpolation function) for the generic power-law hypothesis
    """
    return 10**(lMst_pw(np.log10(eta), Gamma, delta))


# # Master-function arguments 


# # # Energy-Frequency relationship for synchrotron emission  


def Enu(B, nu):
    """
    Typical particle energy in GeV for synchrotron radiation at the frequency nu in GHz and for a magnetic field B in :math:`\\mu`\G

    :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
    :param nu: frequency in GHz

    :return: Particle energy in GeV.
    """
    return eps0 * np.sqrt(nu / B)

# # Dimensionless Syrovatskii variable


def eta(E, B, tau, delta):
    """
    :math:`\\eta` variable as a function of the CRE's energy, magnetic field, tau and delta parameters 

    :param E: CRE energy in GeV
    :param B: magnetic field strength in µG
    :param tau: diffusion time-scale parameter for a 1 GeV CRE in s
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy

    :return: :math:`\\eta` variable
    """
    d = delta_float(delta)
    return np.pi**2 / tau / b0 / (1 + (B / Bc)**2) / (1 - d) / E**(1 - d)
