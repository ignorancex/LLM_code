from dataclasses import dataclass
import numpy as np

# Logging
import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
# Printing to stdout
LOGGER.addHandler(logging.StreamHandler())


# =====================================
# Basic QCD-related utilities
# =====================================

# ---------------------------------
# QCD
# ---------------------------------
# QCD Coupling at the Z boson mass
ALPHAS_MZ = 0.1185

# Number of active quark flavors
N_F = 5.

# Group theory factors
N_C = 3
CF = (N_C**2 - 1)/(2*N_C)
CA = N_C
TF = 1./2.

# QCD Beta function
beta_0 = (11*CA - 4*TF*N_F)/(12 * np.pi)


def alpha_s(mu, freeze_at=None):
    """1-loop strong force coupling. Argument in GeV."""
    alpha_s = ALPHAS_MZ/(1 + 2*ALPHAS_MZ*beta_0*np.log(mu/M_Z))
    if freeze_at:
        alpha_s[alpha_s < 0] = freeze_at
        return np.clip(np.nan_to_num(alpha_s,
                                     freeze_at, posinf=freeze_at),
                       a_max=freeze_at, a_min=None)

    return alpha_s


# ---------------------------------
# Mass scales
# ---------------------------------

# Mass Scales (in GeV)
M_Z = 91.19
M_W = 80.38
M_t = 172.76
Lambda_QCD = .245

mass_val = {'qcd' : Lambda_QCD,
             'z'   : M_Z,
             'w'   : M_W,
             'top' : M_t}

scale_name = {'qcd' : r"$\Lambda_{\rm QCD}$",
              'z'   : r"$m_Z$",
              'w'   : r"$m_W$",
              'top' : r"$m_t$"}

scale_col = {'qcd' : 'rebeccapurple',
             'z'   : 'cadetblue',
             'w'   : 'firebrick',
             'top' : 'peru'}


if __name__ == "__main__":
    Q = np.array([1000])

    LOGGER.info("# ---------------------------------")
    LOGGER.info("# QCD Factors")
    LOGGER.info("# ---------------------------------")
    LOGGER.info(f"At Q = {Q} GeV, alpha_s = {alpha_s(Q)}")
    LOGGER.info(f"as(Q) = {alpha_s(Q) / (4*np.pi) = }")
    LOGGER.info("")
    LOGGER.info(f"Number of Colors: N_C = {N_C}")
    LOGGER.info(f"Number of Flavors: N_F = {N_F}")
    LOGGER.info("")
    LOGGER.info("# ---------------------------------")
    LOGGER.info("# Group Theory Factors")
    LOGGER.info("# ---------------------------------")
    LOGGER.info(f"Fundamental Casimir: CF = {CF}")
    LOGGER.info(f"Adjoint Casimir: CA = {CA}")
    LOGGER.info(f"Fundamental Dynkin Index: TF = {TF}")
    LOGGER.info(f"QCD Beta Function Coefficient: beta_0 = {beta_0}")
    LOGGER.info("")
    LOGGER.info("# ---------------------------------")
    LOGGER.info("# Mass Scales")
    LOGGER.info("# ---------------------------------")
    for scale in mass_val:
        LOGGER.info(f"{scale_name[scale]} = {mass_val[scale]} GeV")
