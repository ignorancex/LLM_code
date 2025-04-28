from diffsph.pyflux import *
from diffsph.utils.consts import *

gauss_convers = 2 / np.pi / (2/5)**2
tophat_twosigma_number = np.sqrt( fsolve(lambda th : sp.gammainc(1/2,  th / 2 ) - 0.9, 2 )[0] )

# ################# WIMPS 

def sigmav_gausslim(nu, a_fit, sigma_fit, beam_size, galaxy, rad_temp, D0 = 3e28, delta = 'kol', B = 2, mchi = 50, channel = 'mumu', self_conjugate = True, manual = False, **kwargs):
    """
    Maximum WIMP self-annihilation cross-section allowed by the exclusion of a Gaussian-shaped signal
        
    .. math::
        a_\\text{fit}\\exp\\left(-\\frac{\\theta^2}{2\\sigma_\\text{fit}^2}\\right)
        
    :param nu: frequency in GHz
    :param a_fit: fitted gaussian amplitude in :math:`\\mu` Jy / beam
    :param sigma_fit: width parameter of the Gaussian template in arcmin
    :param beam_size: beam size in arcseconds
    :param str galaxy: name of the galaxy
    :param str rad_temp: dark matter halo model (``'NFW'``, ``'Einasto'``, etc.)
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G)
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
        
    Keyword arguments
    
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile 

    :return: upper limit for the WIMP self-annihilation cross-section in cm :math:`{}^3` /s
    :rtype: float
    """
    conds = [(channel == 'bb' and mchi <= 4), 
             (channel == 'WW' and mchi <= 80), 
             (channel == 'ZZ' and mchi <= 91),
             (channel == 'hh' and mchi <= 125),
             (channel == 'nunu' and mchi <= 125),
             (channel == 'tt' and mchi <= 173),
             mchi <= 1
            ]
    if any(conds):
        return None
    
    rhalf = gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    ratio = np.sin(dist * sigma_fit / .4 / rad_to_arcmin) / rhalf
    beamsr = np.pi / 4 / np.log(2) * (beam_size / 3600 * np.pi / 180) ** 2
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv'} )
    rargs = {rk: kwargs[rk] for rk in rkeys}
    
    f0 = 2 * sp.sici(np.pi)[0]    # f(n=1,0) is equal to 2 * sp.sici(np.pi)[0]
    


    return f0 * a_fit / gauss_convers / synch_brightness_approx(
        0., nu, galaxy, rad_temp, 'wimp', ratio, D0, delta, B, 'B', manual, mchi = mchi, channel = channel, sv = 1, 
        self_conjugate = self_conjugate, **rargs
    ) / beamsr / 1e6


def sigmav_limest(nu, rms_noise, beam_size, galaxy, rad_temp, ratio = 1, D0 = 3e28, delta = 'kol', B = 2, mchi = 50, channel = 'mumu', self_conjugate = True, manual = False, high_res = False, accuracy = 1, **kwargs):
    """
    (Estimated) maximum WIMP self-annihilation cross-section given the rms noise level of an observation
        
    :param nu: frequency in GHz
    :param rms_noise: RMS noise level of the observation in :math:`\\mu` Jy / beam
    :param beam_size: beam size in arcseconds
    :param str galaxy: name of the galaxy
    :param str rad_temp: dark matter halo model (``'NFW'``, ``'Einasto'``, etc.)
    :param ratio: ratio between the diffusion halo and half-light radii
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G)
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)
        
    Keyword arguments
    
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    
    :return: Estimated upper limit on WIMP self-annihilation cross-section in cm :math:`{}^3` /s
    :rtype: float
    """
    conds = [(channel == 'bb' and mchi <= 4), 
             (channel == 'WW' and mchi <= 80), 
             (channel == 'ZZ' and mchi <= 91),
             (channel == 'hh' and mchi <= 125),
             (channel == 'nunu' and mchi <= 125),
             (channel == 'tt' and mchi <= 173),
             mchi <= 1
            ]
    if any(conds):
        return None
    
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    
    theta_h = np.arcsin( rh / dist)

    beamsr = np.pi / 4 / np.log(2) * (beam_size / 3600 * np.pi / 180) ** 2
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv'} )
    rargs = {rk: kwargs[rk] for rk in rkeys}
    
    hf_ang_radius = hfd(
        synch_flux_density, 
        rad_to_arcmin * theta_h, nu, galaxy, rad_temp, 'wimp', ratio, D0, delta, B, manual, high_res, accuracy, 
        mchi = mchi, channel = channel, sv = 1, self_conjugate = self_conjugate, **rargs
                       ) / 2 / rad_to_arcmin

    Omega_half = np.pi * hf_ang_radius ** 2
    
    Omega_full = np.pi * theta_h **2 / beamsr
    
    Nbeams = Omega_half / beamsr
    
    return tophat_twosigma_number * rms_noise * Omega_full / synch_flux_density(
        rad_to_arcmin * theta_h, nu, galaxy, rad_temp, 'wimp', ratio, D0, delta, B, manual, 
        mchi = mchi, channel = channel, sv = 1, self_conjugate = self_conjugate, **rargs
    ) / np.sqrt( Nbeams)


# ################# DM DECAY


def decay_rate_gausslim(nu, a_fit, sigma_fit, beam_size, galaxy, rad_temp, D0 = 3e28, delta = 'kol', B = 2, mchi = 50, channel = 'mumu', manual = False, **kwargs):
    """
    Maximum dark matter decay rate allowed by the exclusion of a Gaussian-shaped signal
            
    .. math::
        a_\\text{fit}\\exp\\left(-\\frac{\\theta^2}{2\\sigma_\\text{fit}^2}\\right)
        
    :param nu: frequency in GHz
    :param a_fit: fitted gaussian amplitude in :math:`\\mu` Jy / beam
    :param sigma_fit: width parameter of the Gaussian template in arcmin
    :param beam_size: beam size in arcseconds
    :param str galaxy: name of the galaxy
    :param str rad_temp: dark matter halo model (``'NFW'``, ``'Einasto'``, etc.)
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G)
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
        
    Keyword arguments
    
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile 
    :param sigmav: velocity dispersion in km/s for the isothermal sphere :py:func:`diffsph.profiles.templates.sis`
    
    :return: upper limit on the DM decay rate in 1/s
    :rtype: float
    """
    conds = [(channel == 'bb' and mchi <= 8.1), 
             (channel == 'WW' and mchi <= 161), 
             (channel == 'ZZ' and mchi <= 183),
             (channel == 'hh' and mchi <= 251),
             (channel == 'nunu' and mchi <= 251),
             (channel == 'tt' and mchi <= 174 * 2),
             mchi <= 2.5
            ]
    if any(conds):
        return None    
    
    rhalf = gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    ratio = np.sin(dist * sigma_fit / .4 / rad_to_arcmin) / rhalf
    beamsr = np.pi / 4 / np.log(2) * (beam_size / 3600 * np.pi / 180) ** 2
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'width'} )
    rargs = {rk: kwargs[rk] for rk in rkeys}
    
    f0 = 2 * sp.sici(np.pi)[0]    # f(n=1,0) is equal to 2 * sp.sici(np.pi)[0]

    return f0 * a_fit / gauss_convers / synch_brightness_approx(
        0., nu, galaxy, rad_temp, 'decay', ratio, D0, delta, B, 'B', manual, mchi = mchi, channel = channel, width = 1, **rargs
    ) / beamsr / 1e6


def decay_rate_limest(nu, rms_noise, beam_size, galaxy, rad_temp, ratio = 1, D0 = 3e28, delta = 'kol', B = 2, mchi = 50, channel = 'mumu', manual = False, high_res = False, accuracy = 1, **kwargs):
    """
    (Estimated) maximum dark matter decay rate given the rms noise level of an observation
        
    :param nu: frequency in GHz
    :param rms_noise: RMS noise level of the observation in :math:`\\mu` Jy / beam
    :param beam_size: beam size in arcseconds
    :param str galaxy: name of the galaxy
    :param str rad_temp: dark matter halo model (``'NFW'``, ``'Einasto'``, etc.)
    :param ratio: ratio between the diffusion halo and half-light radii
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G)
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)
        
    Keyword arguments
    
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param sigmav: velocity dispersion in km/s for the isothermal sphere :py:func:`diffsph.profiles.templates.sis`
    
    :return: Estimated upper limit on the DM decay rate in 1/s
    :rtype: float
    """
    conds = [(channel == 'bb' and mchi <= 8), 
             (channel == 'WW' and mchi <= 160), 
             (channel == 'ZZ' and mchi <= 182),
             (channel == 'hh' and mchi <= 250),
             (channel == 'nunu' and mchi <= 250),
             (channel == 'tt' and mchi <= 346),
             mchi <= 2
            ]
    if any(conds):
        return None
    
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    
    theta_h = np.arcsin( rh / dist)

    beamsr = np.pi / 4 / np.log(2) * (beam_size / 3600 * np.pi / 180) ** 2
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'width'} )
    rargs = {rk: kwargs[rk] for rk in rkeys}
    
    hf_ang_radius = hfd(
        synch_flux_density, 
        rad_to_arcmin * theta_h, nu, galaxy, rad_temp, 'decay', ratio, D0, delta, B, manual, high_res, accuracy, 
        mchi = mchi, channel = channel, width = 1, **rargs
                       ) / 2 / rad_to_arcmin

    Omega_half = np.pi * hf_ang_radius ** 2
    
    Omega_full = np.pi * theta_h **2 / beamsr
    
    Nbeams = Omega_half / beamsr
    
    return tophat_twosigma_number * rms_noise * Omega_full / synch_flux_density(
            rad_to_arcmin * theta_h, nu, galaxy, rad_temp, 'decay', ratio, D0, delta, B, manual, high_res, accuracy, 
            mchi = mchi, channel = channel, width = 1, **rargs
        ) / np.sqrt( Nbeams)


def generic_rate_gausslim(nu, a_fit, sigma_fit, beam_size, galaxy, rad_temp, D0 = 3e28, delta = 'kol', B = 2, Gamma = 2, **kwargs):
    """
    Maximum CRE production rate (generic power-law hypothesis) allowed by the exclusion of a Gaussian-shaped signal
             
    .. math::
        a_\\text{fit}\\exp\\left(-\\frac{\\theta^2}{2\\sigma_\\text{fit}^2}\\right)
        
    :param nu: frequency in GHz
    :param a_fit: fitted gaussian amplitude in :math:`\\mu` Jy / beam
    :param sigma_fit: width parameter of the Gaussian template in arcmin
    :param beam_size: beam size in arcseconds
    :param str galaxy: name of the galaxy
    :param str rad_temp: dark matter halo model (``'NFW'``, ``'Einasto'``, etc.)
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G)
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`, default value = 2)
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
        
    Keyword arguments
    
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param sigmav: velocity dispersion in km/s for the isothermal sphere :py:func:`diffsph.profiles.templates.sis`
    
    :return: upper limit on the generic CRE production rate in 1/s
    :rtype: float
    """
    
    rhalf = gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    ratio = np.sin(dist * sigma_fit / .4 / rad_to_arcmin) / rhalf
    beamsr = np.pi / 4 / np.log(2) * (beam_size / 3600 * np.pi / 180) ** 2
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'Gamma', 'rate'} )
    rargs = {rk: kwargs[rk] for rk in rkeys}
    
    f0 = 2 * sp.sici(np.pi)[0]    # f(n=1,0) is equal to 2 * sp.sici(np.pi)[0]

    return f0 * a_fit / gauss_convers / synch_brightness_approx(
        0., nu, galaxy, rad_temp, 'generic', ratio, D0, delta, B, 'B', manual = True, Gamma = Gamma, rate = 1, 
         **rargs
    ) / beamsr / 1e6
 

def generic_rate_limest(nu, rms_noise, beam_size, galaxy, rad_temp, ratio = 1, D0 = 3e28, delta = 'kol', B = 2, Gamma = 2, 
                        high_res = False, accuracy = 1, **kwargs):
    """
    (Estimated) maximum CRE production rate (generic power-law hypothesis) given the rms noise level of an observation
                    
    :param nu: frequency in GHz
    :param rms_noise: RMS noise level of the observation in :math:`\\mu` Jy / beam
    :param beam_size: beam size in arcseconds
    :param str galaxy: name of the galaxy
    :param str rad_temp: dark matter halo model (``'NFW'``, ``'Einasto'``, etc.)
    :param ratio: ratio between the diffusion halo and half-light radii
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G)
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`, default value = 2)
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)
        
    Keyword arguments
    
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param sigmav: velocity dispersion in km/s for the isothermal sphere :py:func:`diffsph.profiles.templates.sis`

    :return: Estimated upper limit on the generic CRE production rate in 1/s
    :rtype: float
    """
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    
    theta_h = np.arcsin( rh / dist)

    beamsr = np.pi / 4 / np.log(2) * (beam_size / 3600 * np.pi / 180) ** 2
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'Gamma', 'rate'} )
    rargs = {rk: kwargs[rk] for rk in rkeys}
    
    hf_ang_radius = hfd(
        synch_flux_density, 
        rad_to_arcmin * theta_h, nu, galaxy, rad_temp, 'generic', ratio, D0, delta, B, True, high_res, accuracy, 
        Gamma = Gamma, rate = 1, **rargs
                       ) / 2 / rad_to_arcmin

    Omega_half = np.pi * hf_ang_radius ** 2
    
    Omega_full = np.pi * theta_h **2 / beamsr
    
    Nbeams = Omega_half / beamsr
    
    return tophat_twosigma_number * rms_noise * Omega_full / synch_flux_density(
        rad_to_arcmin * theta_h, nu, galaxy, rad_temp, 'generic', ratio, D0, delta, B, True, high_res, accuracy, 
        Gamma = Gamma, rate = 1, **rargs
    ) / np.sqrt( Nbeams)
    