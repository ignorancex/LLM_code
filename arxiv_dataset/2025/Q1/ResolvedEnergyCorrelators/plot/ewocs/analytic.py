import numpy as np

from numpy import einsum

# Local utilities for QCD
from qcd.qcd_basics import alpha_s, CF, CA, N_F
from qcd.dglap_moments import \
    moment_evolution_matrix, inclusive_evolved_splitting,\
    mass_limited_splitting
    # TODO: remove mass_limited_splitting


# Notes:
#     * The default accuracy used in plotting the functions below
#       is stored in lfm_tools.plot_ewocs

# TODO:     * Unify order of arguments between functions
# TODO:     * Add equations to docstrings

# TODO: MORE SERIOUS -- mu should include Rjet, and doesn't yet


def ewoc_ll(initial_scale, split_scale, final_scale,
            angular_scale, jacobian,
            weights=None,
            initial_flavor='quark'):
    """Leading-logarithmic EWOC."""
    if weights is None:
        n1, n2 = 1.0, 1.0
    else:
        n1, n2 = weights

    # The initial evolution has an index involving the scale
    # and an two indices for the initial/intermediate flavor
    # Indices: aij
    initial_evolution = moment_evolution_matrix(initial_scale,
                                                split_scale,
                                                n1 + n2)

    # The inclusive (summed over final flavors) splitting function
    # has an index for the scale and an index for the intermediate
    # flavor
    # Indices: aj
    inclusive_splitting = inclusive_evolved_splitting(split_scale,
                                                      final_scale,
                                                      (n1, n2))

    # The prefactors involve scale and so have an index "a"
    a_eff = alpha_s(split_scale) / (4 * np.pi)

    prefactor = 2 * a_eff * jacobian / (angular_scale**2.)

    # The indices denote:
    #     a: the scale (accounting for multiple splitting scales)
    #     i: the original partonic flavor
    #     j: the flavor of the intermediate parton
    flavored_ewoc = einsum('a,aij,aj->ai',
                           prefactor,
                           initial_evolution,
                           inclusive_splitting)

    if initial_flavor in ['quark', 'q']:
        return flavored_ewoc[:,0]
    if initial_flavor in ['gluon', 'g']:
        return flavored_ewoc[:,1]

    raise ValueError("Invalid initial partonic flavor " + str(initial_flavor))


# =====================================
# Final Results:
# =====================================
def subjet_eec(var, mu, r_sub, var_name,
               accuracy=None,
               energy=None,
               weights=None,
               R_jet=None):
    """Subjet EEC; takes a particularly
    simple form, as the EEC is simply zero when
    the angular variable is less than r_sub
    """
    if accuracy is None or accuracy == 'fixed':
        accuracy = 'nnlo'
    elif accuracy == 'logarithmic':
        accuracy = 'll'

    # Setting up variable -- using z and converting
    # from cos(theta) if necessary
    # DEBUG -- jacobian factors?
    if var_name == 'z':
        z = var
        jacobian = 1
    if var_name == 'theta':
        z = var**2/4
        jacobian = var/2
    if var_name == 'theta2':
        z = var/4
        jacobian = 1/4
    if var_name == 'costheta':
        z = (1-var)/2
        jacobian = 1/2

    if hasattr(z, '__len__'):
        # typecasting
        z = np.array(z).astype(float)

    # Angles -- will be enforced to be less than
    # the subjet radius
    theta = np.arccos(1 - 2 * z)

    # Unifying existing accuracy options
    if accuracy in ['lo', 'nlo', 'nnlo']:
        assert energy is None, "Fixed-order EEC does not support energy dependence"
        assert weights is None, "Fixed-order EEC does not support weights"
        assert R_jet is None, "Fixed-order EEC does not support jet radius"

        if accuracy == 'lo':
            eec = eec_lo
        elif accuracy == 'lo+ll':
            raise NotImplementedError("LO + LL for EEC not yet "
                                      "implemented.")
        elif accuracy == 'nlo':
            eec = eec_nlo
        elif accuracy == 'nnlo':
            eec = eec_nnlo

        return jacobian * np.array(eec(z, mu)) * np.greater(theta, r_sub)

    # Allowing a greater number of parameters
    if energy is None:
        energy = 2*mu
    if R_jet is None:
        R_jet = 1.0

    if accuracy == 'll':
        eec = eec_ll
    else:
        raise ValueError("Invalid accuracy " + str(accuracy))

    eec_array = np.array(eec(z, mu, weights, R_jet, r_sub))

    return eec_array * np.greater(theta, r_sub) * jacobian


def m2_subjet_ewoc(m2, mu, r_sub, energy,
                   accuracy=None,
                   m2_cut=None,
                   weights=None,
                   R_jet=None):
    """Mass-squared EWOC."""
    if accuracy is None or accuracy.lower() == 'fixed':
        accuracy = 'lo'
    elif accuracy.lower() == 'logarithmic':
        accuracy = 'll'
    elif isinstance(accuracy, str):
        accuracy = accuracy.lower()

    if hasattr(m2, '__len__'):
        # typecasting
        m2 = np.array(m2).astype(float)

    # Looping over existing accuracy options
    if accuracy == 'lo':
        assert weights is None, "LO EWOC does not support weights"
        if m2_cut is None:
            assert r_sub is not None, "Must specify r_sub for "\
                    "LO EWOC if m2_cut is not specified"

        if m2_cut is None and R_jet is None:
            m2crit = (energy * r_sub / 4)**2
            def m2_ewoc(m2prime, muprime):
                # Implement in a way that is numpy-friendly
                # returning subcritical EWOC below m2crit
                # and the regular LO EWOC above:
                return np.greater(m2prime, m2crit) * m2_ewoc_lo(m2prime,
                                                                muprime) + \
                    np.less_equal(m2prime, m2crit) * \
                      subcritical_m2_subjet_ewoc(m2prime, muprime,
                                                 r_sub, energy)
            return np.array(m2_ewoc(m2, mu))

        elif m2_cut is None and R_jet is not None:
            return np.array(m2_jet_ewoc(m2, mu, R_jet, r_sub, energy,
                                        augment_ll=False))

        elif r_sub is None:
            assert m2_cut is not None, "Must specify m2_cut for LO EWOC "\
                    "if r_sub is not specified"
            return np.array(m2_ewoc_lo(m2, mu)) * np.greater(m2, m2_cut)

        else:
            raise ValueError("Must specify either m2_cut or r_sub for LO EWOC")

    elif accuracy == 'lo+ll':
        return np.array(m2_jet_ewoc(m2, mu, R_jet, r_sub, energy,
                                    augment_ll=True))

    # Allowing a greater number of parameters
    if m2_cut is None:
        m2_cut = 0
    if energy is None:
        energy = 2*mu
    # DEBUG: Remove this if statement and support R_jet in general
    if R_jet is None:
        R_jet = 1.0

    if accuracy == 'll':
        ewoc = m2_ewoc_ll
    else:
        raise ValueError("Invalid accuracy " + str(accuracy))

    ewoc_array = np.array(ewoc(m2, mu, weights, R_jet, r_sub))

    return ewoc_array


# =====================================
# No jet or subjet radius
# =====================================

# ---------------------------------
# EECs
# ---------------------------------
def eec_lo(z, mu):
    """LO EEC defined in terms of the variable
    z = (1 - cos(theta))/2

    No subjet radius included, since the LO EEC
    is simply zero for theta < rsub.
    """
    a_s = alpha_s(mu) / (4 * np.pi)
    # Not quite right yet, need to run at higher acccuracy

    z = z.astype(float)

    return (2.*a_s)/(z*(1-z))

def eec_nlo(z, mu):
    """NLO EEC defined in terms of the variable
    z = (1 - cos(theta))/2

    No subjet radius included, since the NLO EEC
    is simply zero for theta < rsub.
    """
    a_s = alpha_s(mu) / (4 * np.pi)
    # Not quite right yet, need to run at higher acccuracy

    z = z.astype(float)

    nlo_piece  = -11.5333*np.log(z) + 81.4809

    return (2.*a_s + a_s**2.*nlo_piece)/(z*(1-z))

def eec_nnlo(z, mu):
    """NNLO EEC defined in terms of the variable
    z = (1 - cos(theta))/2

    No subjet radius included, since the NNLO EEC
    is simply zero for theta < rsub.
    """
    a_s = alpha_s(mu) / (4 * np.pi)
    # Not quite right yet, need to run at higher acccuracy

    z = z.astype(float)

    nlo_piece  = -11.5333*np.log(z) + 81.4809
    nnlo_piece = 45.1489*np.log(z)**2. - 1037.73*np.log(z) + 2871.36

    return (2.*a_s + a_s**2.*nlo_piece + a_s**3.*nnlo_piece)/(z*(1-z))

def eec_ll(z, mu, weights,
           Rjet, r_sub):
    """Leading-logarithmic EEC.

    No subjet included, since the LL EEC is
    simply zero for theta < r_sub
    """
    if weights is None:
        weights = (1.0, 1.0)

    initial_scale = mu * Rjet

    theta = np.arccos(1 - 2 * z)
    split_scale = mu * theta

    final_scale = mu * r_sub

    # MATH: I'm not sure what the EEC jacobian should be
    # DEBUG: Maybe it should be 2 T.T
    jacobian=4

    return ewoc_ll(initial_scale, split_scale,
            final_scale=final_scale,
            angular_scale=theta,
            jacobian=jacobian,
            weights=weights)


# ---------------------------------
# EWOCs
# ---------------------------------
def m2_ewoc_lo(m2, mu):
    """Leading order particle-level mass-squared EWOC.
    Of course, does not include subjet or jet radius effects."""
    a_s = alpha_s(mu) / (4 * np.pi)
    return a_s * CF * 3 / (4 * m2)


def subcritical_m2_subjet_ewoc(m2, mu, r_sub, energy):
    """Subjet mass-squared EWOC below the
    critical mass-squared value
        (energy * r_sub / 4)**2
    set by the subjet radius.

    Does not include jet radius effects.
    ."""
    a_s = alpha_s(mu) / (4 * np.pi)
    numerator = (
        (152 * energy**2 * r_sub**2) / np.sqrt(energy**2 * r_sub**2 - 16 * m2) +
        16 * np.sqrt(energy**2 * r_sub**2 - 16 * m2) -
        (128 * m2) / np.sqrt(energy**2 * r_sub**2 - 16 * m2) +
        (9 * energy**3 * r_sub**3 * (-((energy * r_sub) / np.sqrt(energy**2 * r_sub**2 - 16 * m2))) - 1) /
        (1 / 8 * energy * r_sub * (np.sqrt(energy**2 * r_sub**2 - 16 * m2) + energy * r_sub) - m2)
    )
    denominator = 12 * energy**3 * r_sub**3

    # Take all nans and turn them into zeros
    result = np.nan_to_num(a_s * CF * numerator / denominator)
    return result


def m2_jet_ewoc(m2, mu, R_jet, r_sub, energy,
                augment_ll):
    assert mu == energy*R_jet/2.,\
            "Inconsistent renormalization scale mu"

    a_s = alpha_s(mu) / (4 * np.pi)

    m2jet = (energy * R_jet / 4)**2.
    m2sub = (energy * r_sub / 4)**2.

    larger_region = np.nan_to_num(np.greater(m2jet, m2)*\
        (1.-1/6.*m2/m2jet)*np.sqrt(1.-m2/m2jet))

    smaller_region = np.nan_to_num(np.greater(m2sub, m2)*\
        (1.-1/6.*m2/m2sub)*np.sqrt(1.-m2/m2sub))

    larger_region = np.nan_to_num(larger_region)

    lo_result = 3*a_s*CF*(larger_region - smaller_region)/m2

    # Additional scaling inserted by hand
    # obtained from LL computation
    if augment_ll:
        a_s = alpha_s(np.sqrt(m2)) / (4 * np.pi)

        # coefficients
        f_1 = np.sqrt((84.*CA - 125.*CF)**2.\
                      + 160.*(21.*CA-19.*CF)*N_F\
                      + 400.*N_F**2.)
        f_2 = 55.*CF - 20.*N_F

        # exponents
        kappa_1 = 2.*a_s*f_1/15.
        kappa_2 = -a_s*(84.*CA + 125.*CF + 20.*N_F + f_1)/15.

        ll_factor = \
                (m2/mu**2.)**(kappa_2/2.) * \
                (
                     (f_1 + f_2)
                     +
                     84*CA*((m2/mu**2.)**(kappa_1/2.) - 1.)\
                     +
                     (f_1-f_2)*(m2/mu**2.)**(kappa_1/2.)\
                ) / (2*f_1)
        # return ll_factor/(2*m2)  # DEBUG: to visualize
        # return lo_result * ll_factor**6./5.
        # print(f'{m2=}')
        # print(f'{np.log(ll_factor)/np.log(m2)=}')
        return lo_result * ll_factor

    return lo_result


def m2_ewoc_ll(m2, mu, weights, Rjet, rsub,
               initial_flavor='quark',
               well_separated=True):
    """Leading-logarithmic EEC.

    No subjet included, since the LL EEC is
    simply zero for theta < r_sub

    The answer is simpler in the case where all scales are well-separated,
    Q=mu >> Q*Rjet >> m >> Q*rsub
    """
    assert well_separated, "Don't have analytic control unless all "\
            "scales are well separated"
    if weights is None:
        n1, n2 = 1.0, 1.0
    else:
        n1, n2 = weights

    # The initial evolution has an index involving the scale
    # and an two indices for the initial/intermediate flavor
    # Indices: aij
    # MATH: Assuming similar fragmentation behavior
    # MATH:     between angle and mass
    # initial_evolution = mass_limited_moment_evolution(
    #                         m2, mu, n1+n2, Rjet, rsub)
    split_scale = np.maximum(np.sqrt(m2), mu * rsub)
    initial_evolution = moment_evolution_matrix(mu * Rjet, split_scale,
                                                n1 + n2)

    # TOGGLE: Printing intitial evolution matrix

    # The inclusive (summed over final flavors) splitting function
    # has an index for the scale and an index for the intermediate
    # flavor
    # Indices: aj
    # MATH:     and therefore putting the burden of the correct
    # MATH:     behavior on the splitting function
    if well_separated:
        # If the scales are well separated, I'll assume that the
        inclusive_splitting = inclusive_evolved_splitting(np.sqrt(m2),
                                                          mu*rsub,
                                                          (n1,n2))
    else:
        # DEPRECATED: An old attempt at getting an LL solution
        exclusive_splitting = mass_limited_splitting(
                                m2, mu, (n1, n2), rsub)
        inclusive_splitting = einsum('ajkl->aj', exclusive_splitting)

    # TOGGLE: Printing inclusive mass-limited splitting function

    # The prefactors involve scale and so have an index "a"
    # MATH: should alpha_s be evaluated at sqrt(m2)?
    a_eff = alpha_s(split_scale) / (4 * np.pi)

    # MATH: factors of 2 from jacobian in EWOC prefactor?
    # prefactor = a_eff / (m2/mu**2.)
    prefactor = a_eff / (m2)

    # The indices denote:
    #     a: the scale (accounting for multiple splitting scales)
    #     i: the original partonic flavor
    #     j: the flavor of the intermediate parton
    flavored_ewoc = einsum('a,aij,aj->ai',
                           prefactor,
                           initial_evolution,
                           inclusive_splitting)

    # Make it zero if not mu*Rjet > m > mu*rsub
    well_separated_theta = np.where(
                        (mu*Rjet > 10*np.sqrt(m2)) * (np.sqrt(m2) > mu*rsub),
                        1, 0)

    # DEBUG
    well_separated_theta=1  # DEBUG: to see if the plot will finally show up
    # print(flavored_ewoc[:,0]*well_separated_theta)

    if initial_flavor in ['quark', 'q']:
        return flavored_ewoc[:,0] * well_separated_theta
    if initial_flavor in ['gluon', 'g']:
        return flavored_ewoc[:,1] * well_separated_theta

    raise ValueError("Invalid initial partonic flavor " + str(initial_flavor))
