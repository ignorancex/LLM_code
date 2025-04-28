import numpy as np
import scipy.special as sc

# Matrix algebra
from numpy import einsum
from scipy.linalg import expm

# Conversion from mathematica expressions
from sympy import var

# QCD basics
from qcd.qcd_basics import alpha_s, TF, CF, CA, N_F, beta_0


def harmonic_number(val: int) -> float:
    """Harmonic number H_n = 1 + 1/2 + ... + 1/n."""
    return sum(1./i for i in range(1, int(val)+1))


# =====================================
# Splitting Function Moments
# =====================================

# TODO: Add docstring with QCD defs

# Jet calculus splitting function moments:
# TODO: fix n here, should be n-1

def inclusive_splitting_moment(n: int):
    """Inclusive splitting function moment:
    """
    # TODO: Equation in docs
    sum_piece = - 2 * (harmonic_number(n+1) - 1)

    a_qq = CF * (-1/2 + 1/((n+1)*(n+2)) + sum_piece)
    a_gq = CF * (n**2 + 3*n + 4)\
                /(n*(n+1)*(n+2))
    a_qg = 2 * TF * N_F * (n**2 + 3*n + 4)\
                         / ((n+1)*(n+2)*(n+3))
    a_gg = CA * (-1/6 + 2/(n*(n+1)) + 2/((n+2)*(n+3))\
                 + sum_piece)\
          - N_F/3

    # Note that the _row_ denote the initial parton
    # and the _column_ denotes the final parton.
    # I'm using some inconsistent notation...
    return np.array([[a_qq, a_gq], [a_qg, a_gg]])


def exclusive_splitting_moment(weights: (int, int)):
    """Exclusive splitting function moment:
    """
    # TODO: Equation in docs
    m, n = weights

    p_q_qg = CF * (sc.beta(m+1, n) + sc.beta(m+3, n))
    p_q_gq = CF * (sc.beta(m, n+1) + sc.beta(m, n+3))

    # MATH: I think I'm missing an effective factor of 2
    # MATH:     because of the sum on final states which
    # MATH:     counts both q qbar and qbar q
    p_g_qq = N_F * (sc.beta(m+3, n+1) + sc.beta(m+1, n+3))
    # MATH/DEBUG: Adding in the factor of 2 by hand:
    # p_g_qq = 2 * N_F * (sc.beta(m+3, n+1) + sc.beta(m+1, n+3))

    p_g_gg = 2*CA * (sc.beta(m, n+2) + sc.beta(m+2, n)\
                    + sc.beta(m+2, n+2))

    # Note that the first index denotes the initial parton
    # and the second two indices denotes the final partons.
    return np.array([
                # a = Initial Quark
                [
                    # c = Final Quark
                    [0, p_q_gq], # b = quark, gluon
                    # c = Final Gluon
                    [p_q_qg, 0] # b = quark, gluon
                ],
                # a = Initial Gluon
                [
                    # c = Final Quark
                    [p_g_qq, 0], # b = quark, gluon
                    # c = Final Gluon
                    [0, p_g_gg] # b = quark, gluon
                ]
    ])


# ---------------------------------
# Mass-Limited Expressions
# ---------------------------------

def mass_limited_splitting(m2, energy, weights, rsub):
    """Right now, only the (1,1) moment of the mass limited splitting
    function, which integrates z(1-z) against the splitting function
    matrix in the region
        z(1-z) < e2/rsub^2
    """
    if isinstance(m2, list) or isinstance(m2, np.ndarray):
        return np.array([mass_limited_splitting(m2_i, energy, weights, rsub)
                         for m2_i in m2])

    if rsub == 0 or rsub is None:
        return exclusive_splitting_moment(weights)

    e2ratio = m2 /(energy**2 * rsub**2)

    if e2ratio >= 1/4:
        return exclusive_splitting_moment(weights)

    assert weights == (1, 1), f"Unsupported {weights =}, "\
        "only weights = (1,1) is supported at the moment."

    sqrt_factor = np.sqrt(1 - 4*e2ratio)

    # (1,1) ONLY:
    p_q_gq = CF*(3*(1-sqrt_factor) + 2*e2ratio*sqrt_factor)/4
    p_q_qg = p_q_gq

    p_g_qq = N_F*(1-sqrt_factor +\
                  2*e2ratio*(2*e2ratio-1)*sqrt_factor)/20
    # MATH/DEBUG: Adding in a factor of 2 by hand (see above):
    # p_g_qq = N_F*(1-sqrt_factor +\
    #               2*e2ratio*(2*e2ratio-1)*sqrt_factor)/10

    p_g_gg = CA * (7*(1-sqrt_factor) -\
                   2*e2ratio*(e2ratio-3)*sqrt_factor)/5

    # Return statement is identical to the unmodified moment above
    return np.array([
                # a = Initial Quark
                [
                    # c = Final Quark
                    [0, p_q_gq], # b = quark, gluon
                    # c = Final Gluon
                    [p_q_qg, 0] # b = quark, gluon
                ],
                # a = Initial Gluon
                [
                    # c = Final Quark
                    [p_g_qq, 0], # b = quark, gluon
                    # c = Final Gluon
                    [0, p_g_gg] # b = quark, gluon
                ]
    ])



# =====================================
# DGLAP Evolution Matrices
# =====================================

def moment_evolution_matrix(initial_scale, final_scale, weight):
    """Evolution of moments from initial to final scale."""
    scale_change = np.log(alpha_s(final_scale) / alpha_s(initial_scale))
    scale_change = scale_change / (2*np.pi*beta_0)

    # For the moment evolution,
    # the first index/row denotes the scale (accounting
    #     for the possibility of multiple values of scale_change)
    # the second index/row denotes the initial parton and
    # the third index/column denotes the final parton.
    scale_change = np.nan_to_num(scale_change, nan=0)
    evolution_matrix = np.array([
        expm(Delta * inclusive_splitting_moment(weight))
        for Delta in scale_change
    ])

    return evolution_matrix


def exclusive_evolved_splitting(initial_scale, final_scale, weights):
    """Evolution of splitting function from initial to final scale,
    where the final states (l and m) below are known."""
    # TODO: Equation in docs
    splitting_factor = exclusive_splitting_moment(weights)

    evolution_1 = moment_evolution_matrix(initial_scale, final_scale, weights[0])
    evolution_2 = moment_evolution_matrix(initial_scale, final_scale, weights[1])

    # The indices denote:
    #     a: the scale change
    #     i: the initial parton
    #     l, m: the two final partons
    return einsum('ijk,ajl,akm->ailm',
                  splitting_factor, evolution_1, evolution_2)


def inclusive_evolved_splitting(initial_scale, final_scale, weights):
    """Evolution of splitting function from initial to final scale,
    where the final states of the splitting are summed over."""
    splitting_factor = exclusive_splitting_moment(weights)

    # If either weights are 1, sum rules allow us to simplify the
    # fully inclusive calculation;
    # the sum over final states after the evolution of the weight-1
    # parton is actually independent of the evolution by energy
    # conservation.
    if weights[0] == 1 and weights[1] == 1:
        return einsum('ijk,a->ai', splitting_factor,
                      np.ones_like(initial_scale * final_scale))

    if weights[0] == 1 and weights[1] != 1:
        evolution_2 = moment_evolution_matrix(initial_scale, final_scale, weights[1])
        return einsum('ijk,akm->ai', splitting_factor, evolution_2)

    if weights[0] != 1 and weights[1] == 1:
        evolution_1 = moment_evolution_matrix(initial_scale, final_scale, weights[0])
        return einsum('ijk,ajl->ai', splitting_factor, evolution_1)

    return einsum('ajkl->aj', exclusive_evolved_splitting(initial_scale,
                                                          final_scale,
                                                          weights))



# =====================================
# Main
# =====================================
if __name__ == "__main__":
    Q = np.array([1000])
    delta = np.array([0.1])

    LOGGER.info("# ---------------------------------")
    LOGGER.info("# QCD Factors")
    LOGGER.info("# ---------------------------------")
    LOGGER.info(f"At Q = {Q} GeV, alpha_s = {alpha_s(Q)}")
    LOGGER.info(f"as(Q) = {alpha_s(Q) / (4*np.pi) = }")
    LOGGER.info(f"Using a final scale Q*delta set by delta = {delta}")
    LOGGER.info("")

    LOGGER.info("# ---------------------------------")
    LOGGER.info("# Splitting Function Moments (Inclusive)")
    LOGGER.info("# ---------------------------------")
    LOGGER.info(f"{inclusive_splitting_moment(2)=}")
    LOGGER.info(f"{((-25*CF/12, 7*CF/12), (7*TF*N_F/15, -7*CA/5 - 2*TF*N_F/3))=}")
    LOGGER.info("")

    LOGGER.info("# ---------------------------------")
    LOGGER.info("# Splitting Function Moments (Exclusive)")
    LOGGER.info("# ---------------------------------")
    LOGGER.info("Before summing over final states:")
    LOGGER.info(f"{exclusive_splitting_moment((1,1))=}")
    LOGGER.info("After summing over final states:")
    summed_splitting_moment_11 = einsum('ijk->i',
                                 exclusive_splitting_moment((1, 1))
                                     )
    LOGGER.info(f'{summed_splitting_moment_11=}')
    LOGGER.info("Expected summed result:")
    LOGGER.info(f'{(3*CF/2, 7*CA/5 + TF*N_F/5)=}')
    LOGGER.info("")

    LOGGER.info("# ---------------------------------")
    LOGGER.info("# Evolution Matrix")
    LOGGER.info("# ---------------------------------")
    LOGGER.info(f"{moment_evolution_matrix(Q, Q*delta, 2)=}")
    LOGGER.info("Mathematica result:")
    LOGGER.info(np.array([
        [0.814045, 0.0634205],
        [0.0422803, 0.646132]
    ]))
    LOGGER.info("These two should be transposes -- our python function uses "\
        "the row to denote the initial parton and the column to denote "\
        "the final parton; the Mathematica result is the reverse.")
    LOGGER.info("")

    LOGGER.info("# ---------------------------------")
    LOGGER.info("# Evolved Splitting Function Moments")
    LOGGER.info("# ---------------------------------")
    summed_evolved_splitting_moment_11 = einsum('aijk->ai',
                        exclusive_evolved_splitting(
                                        Q, Q*delta,
                                        (1, 1))
                                        )
    LOGGER.info(f'{summed_evolved_splitting_moment_11=}')
    LOGGER.info("Mathematica result:")
    LOGGER.info(np.array([
        [0, 0]
    ]))
    LOGGER.info("Makes sense to me that this is the same as the unevolved "\
          "splitting function moment by sum rules, but still need to "\
          "find a way to check this more easily in Mathematica.")
