import numpy as np

from qcd.qcd_basics import mass_val, scale_name, scale_col


def obs_to_energy_value(observable, obs_vals, **kwargs):
    """Returns the energy scales associated with the values
    `obs_vals` of the observable with given name `observable`.
    """
    energy = float(kwargs['energy'])

    # Taking pdf effects into account
    for pid in [kwargs['PID_1'],
                kwargs['PID_2']]:
        if pid == 2212:
            energy /= np.sqrt(13)

    # Rough estimates of the energies of softer/harder emissions
    expected_hard_energy = energy/3
    expected_soft_energy = energy/10

    # Observables
    if observable in ['jet_mass',
                      'mass_onshell',
                      'mass_approx',
                      'mass_tot']:
        return obs_vals
    elif observable in ['m2_onshell',
                        'm2_approx',
                        'm2_tot']:
        return np.sqrt(obs_vals)
    elif observable == 'costheta':
        return energy * np.sqrt((1.-obs_vals)/8.)
    elif observable == 'theta':
        return energy * obs_vals/4
    elif observable == 'deltaR':
        return energy * obs_vals/4
    elif observable == 'theta2':
        return energy * np.sqrt(obs_vals)/4
    elif observable == 'z':
        return energy * np.sqrt(obs_vals)/2
    elif observable == 'formtime':
        return np.sqrt(expected_hard_energy/obs_vals)
    elif observable == 'kt':
        return energy * obs_vals / (expected_soft_energy * 2 * 4.5)
    elif observable in ['e1', 'splite1']:
        return energy**2. * (obs_vals / expected_soft_energy)**2. / 2.
    elif observable in ['e2', 'splite2']:
        return energy**2. * obs_vals / (2. * expected_soft_energy)
    elif observable in ['e4', 'splite4']:
        return energy**2. * np.sqrt(obs_vals/expected_soft_energy) / 2.
    elif observable in ['e6', 'splite6']:
        return energy**2. * np.power(obs_vals/expected_soft_energy,
                                     1/3.) / 2.

    raise AssertionError(f"Invalid {observable = }")


def mask_obs_by_region(observable, obs_vals, region_name, **kwargs):
    """Returns a mask for arrays which is true when the given obs_vals
    are in the named region for the observable.
    """
    # ``Region'' means one of the following:
    valid_regions = ['non-perturbative', 'resummed', 'fixed order']
    if region_name not in valid_regions:
        raise ValueError(f"Invalid region name {region_name}, "\
                         f"must be one of {valid_regions}.")

    # Getting the associated energy scales and energy of the collision
    energy = float(kwargs['energy'])
    energy_vals = obs_to_energy_value(observable, obs_vals, **kwargs)

    # Masks for each region
    if region_name == 'non-perturbative':
        mask = (energy_vals <= 1.)
    elif region_name == 'resummed':
        mask = np.logical_and(10<=energy_vals,
                  energy_vals<energy**2./40000.)  # DEBUG: first pass
    elif region_name == 'fixed order':
        mask = (energy**2./10000 < energy_vals)  # DEBUG: first pass
    else:
        raise AssertionError(f"Dealing with unknown {region_name=}.")


def region_bounds(observable, region_name, **kwargs):
    """Returns the bounds on the given region
    for the given observable.
    """
    energy = float(kwargs['energy'])
    jet_radius = float(kwargs['jet_radius'])

    # Taking pdf effects into account
    for pid in [kwargs['PID_1'],
                kwargs['PID_2']]:
        if pid == 2212:
            energy /= np.sqrt(13)

    # Rough estimates of the energies of softer/harder emissions
    expected_hard_energy = energy/3
    expected_soft_energy = energy/10

    # energy vals greater than 10 and less than E^2/40000
    # DEBUG: first pass

    fudge = 1/40000

    if region_name == 'resummed':
        # Saying for now that the resummed region is
        # set by subjet radius and jet radius
        if observable in mass_observables or\
                observable in m2_observables:
            # E_low = energy * subjet_radius / 4  # DEBUG: Ideal
            E_low = 10
            E_high = .38 * energy * jet_radius / 4
        else:
            # This is a bit silly but should work for
            # angle-based observables, I guess
            E_low = 10.
            E_high = energy * jet_radius / 4
            # E_high = energy**2. * fudge  # DEBUG: first pass
    elif region_name == 'non-perturbative':
        E_low = 1e-50
        E_high = 1
    else:
        raise ValueError(f"Invalid {region_name = }")

    # Observables
    if observable in mass_observables:
        return (E_low, E_high)
    elif observable in m2_observables:
        return (E_low**2., (E_high)**2)
    elif observable == 'costheta':
        return (1 - 8*(E_high/energy)**2., 1 - 8*(E_low/energy)**2)
    elif observable == 'theta':
        return (4*E_low/energy, 4*E_high/energy)
    elif observable == 'deltaR':
        return (4*E_low/energy, 4*E_high/energy)
    elif observable == 'theta2':
        return ((4*E_low/energy)**2., (4*E_high/energy)**2.)
    elif observable == 'z':
        return ((2*E_low/energy)**2., (2*E_high/energy)**2.)
    elif observable == 'formtime':
        return (expected_hard_energy/E_high**2.,
                expected_hard_energy/E_low**2.)
    elif observable == 'kt':
        return (expected_soft_energy/energy * 2. * 4.5 * E_low,
            expected_soft_energy/energy * 2. * 4.5 * E_high)

    # DEBUG: I'm bothered by inconsistency of units here:
    elif observable in ['e1', 'splite1']:
        return (expected_soft_energy*(2*E_low/energy**2.)**(1./2.),
            expected_soft_energy*(2*E_high/energy)**(1./2.))
    elif observable in ['e2', 'splite2']:
        return (expected_soft_energy*(2*E_low/energy**2.),
            expected_soft_energy*(2*E_high/energy))
    elif observable in ['e4', 'splite4']:
        return (expected_soft_energy*(2*E_low/energy**2.)**2.,
            expected_soft_energy*(2*E_high/energy)**2.)
    elif observable in ['e6', 'splite6']:
        return (expected_soft_energy*(2*E_low/energy**2.)**3.,
            expected_soft_energy*(2*E_high/energy)**3.)

    raise AssertionError(f"Invalid {observable = }")


default_moment = 1

def pt_fudge(moment=default_moment):
    if moment == 1:
        # average pt
        return 1.227
    if moment == 2:
        # sqrt(average pt^2)
        return 1.2557

def pt_val_from_min(pt_min, moment=default_moment):
    if pt_min == 500:
        if moment == 1:
            # average pt
            energy = 653.26
        if moment == 2:
            # sqrt(average pt^2)
            energy = 676.34
        if moment == -1:
            # 1/ (average 1/pt)
            energy = 622.12
        if moment == -2:
            # 1/ sqrt(average 1/pt^2)
            energy = 611.26
    elif pt_min == 400:
        if moment == 1:
            # average pt
            energy = 532.07
        if moment == 2:
            # sqrt(average pt^2)
            energy = 553.08
        if moment == -1:
            # 1/ (average 1/pt)
            energy = 503.94
        if moment == -2:
            # 1/ sqrt(average 1/pt^2)
            energy = 494.24
    elif pt_min == 600:
        if moment == 1:
            # average pt
            energy = 766.94
        if moment == 2:
            # sqrt(average pt^2)
            energy = 789.46
        if moment == -1:
            # 1/ (average 1/pt)
            energy = 735.46
        if moment == -2:
            # 1/ sqrt(average 1/pt^2)
            energy = 724.17
    else:
        raise ValueError(f"Untested {pt_min=}.")

    return energy



def expected_obs_peak(observable, **kwargs):
    """Returns the expected peak for the given
    observable.
    """
    try:
        energy = float(kwargs['energy'])
    except:
        return -1.0
    mass = mass_val[kwargs['outstate']]

    # Taking pdf effects into account
    for pid in [kwargs['PID_1'],
                kwargs['PID_2']]:
        if pid == 2212:
            energy /= np.sqrt(13)  # DEBUG: how is this changed by
                                   # DEBUG: the energy of the collision?

    if kwargs['PID_1'] == 2212 and kwargs['PID_2'] == 2212:
        energy = pt_val_from_min(kwargs.get('pt_min', 500))

    # Rough estimates of the energies of softer/harder emissions
    expected_hard_energy = energy/3
    expected_soft_energy = energy/10

    # Observables
    if observable in ['jet_mass',
                      'mmdt',
                      'mass',
                      'mass_onshell',
                      'mass_approx',
                      'mass_tot']:
        peak = mass
    elif observable in ['m2', 'm2_onshell',
                        'm2_approx',
                        'm2_tot']:
        peak = mass**2.
    elif observable == 'costheta':
        peak = 1 - 8.*mass**2./energy**2.
    elif observable == 'theta':
        peak = 4*mass/energy
    elif observable == 'deltaR':
        peak = 2*mass/energy * pt_fudge()
    elif observable == 'theta2':
        peak = (4*mass/energy)**2.
    elif observable == 'z':
        peak = (2*mass/energy)**2.
    elif observable in ['formtime', 'formtime_tot',
                        'formtime_approx',
                        'formtime_onshell']:
        peak = expected_hard_energy/mass**2.
    elif observable == 'kt':
        peak = expected_soft_energy * 2.*mass/energy * 4.5
    elif observable == 'e1':
        peak = expected_soft_energy * np.sqrt(2.*mass/energy**2.)
    elif observable == 'e2':
        peak = expected_soft_energy * (2.*mass/energy**2.)
    elif observable == 'e4':
        peak = expected_soft_energy * (2.* mass/energy**2.)**2.
    elif observable == 'e6':
        peak = expected_soft_energy * (2.* mass/energy**2.)**3.
    elif observable == 'splite1':
        peak = expected_soft_energy * np.sqrt(2.*mass/energy**2.)
    elif observable == 'splite2':
        peak = expected_soft_energy * (2.*mass/energy**2.)
    elif observable == 'splite4':
        peak = expected_soft_energy * (2.* mass/energy**2.)**2.
    elif observable == 'splite6':
        peak = expected_soft_energy * (2.* mass/energy**2.)**3.
    elif isinstance(observable, list):
        if all(obs[:4] == 'mass' or obs == 'jet_mass'
               for obs in observable):
            peak = mass
        elif all(obs[:2] == 'm2' for obs in observable):
            peak = mass**2.
        elif all(len(obs) == 2 and obs[0] == 'e'
                 for obs in observable):
            peak = -1  # DEBUG: No peak shown for angularities
        elif all(len(obs) == 7 and obs[:5] == 'splite'
                 for obs in observable):
            peak = -1  # DEBUG: No peak shown for angularities
        else:
            raise AssertionError(f"Invalid {observable = }")
    else:
        raise AssertionError(f"Invalid {observable = }")

    return peak


def expected_peaks_scales_colors(observable, **kwargs):
    """Returns expected peaks, names of the associated scales,
    and colors for plotting the expected peak for the
    given observable.
    """
    exp_peaks = [expected_obs_peak(observable, **kwargs)]
    scales = [scale_name[kwargs['outstate']]]
    colors = [scale_col[kwargs['outstate']]]

    # If we expect additional peaks from other physics
    if kwargs['outstate'] == 'w':
        exp_peaks.append(expected_obs_peak(observable,
                            **dict(kwargs, outstate='w')))
        scales.append(scale_name['w'])
        colors.append(scale_col['w'])
    if kwargs['outstate'] == 'top':
        exp_peaks.append(expected_obs_peak(observable,
                            **dict(kwargs, outstate='top')))
        scales.append(scale_name['top'])
        colors.append(scale_col['top'])

    return exp_peaks, scales, colors
