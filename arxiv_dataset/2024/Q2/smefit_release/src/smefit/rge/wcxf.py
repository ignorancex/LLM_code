# -*- coding: utf-8 -*-
# Dictionary translating from the smefit basis to the Warsaw basis in the WCxf
import numpy as np

# Values at MZ
alpha_s = 0.118
mw = 80.387
mz = 91.1876
gs = np.sqrt(4 * np.pi * alpha_s)
sw = np.sqrt(1 - mw**2 / mz**2)
cw = np.sqrt(1 - sw**2)


# This creates a dictionary to go from SMEFiT basis to Warsaw.
# In particular, for each operator, it tells you which Wilson coefficients
# need to be switched on in the Warsaw basis and the corresponding values.
# These are basically the equations between operators.
# O_{SMEFiT} = sum_i value_i * O_{Warsaw, i}
wcxf_translate = {
    # Bosonic
    "OWWW": {"wc": ["W"]},
    "OpBox": {"wc": ["phiBox"]},
    "OpD": {"wc": ["phiD"]},
    "OpWB": {"wc": ["phiWB"]},
    "OpG": {"wc": ["phiG"]},
    "OpW": {"wc": ["phiW"]},
    "OpB": {"wc": ["phiB"]},
    "Op": {"wc": ["phi"]},
    # Dipoles
    "OtG": {"wc": ["uG_33"], "value": ["gs"]},
    "OtW": {"wc": ["uB_33", "uW_33"], "value": [cw / sw, 1.0]},
    "OtZ": {"wc": ["uB_33"], "value": [-1.0 / sw]},
    # Quark Currents
    "O3pq": {"wc": ["phiq1_11", "phiq1_22", "phiq3_11", "phiq3_22"]},
    "OpqMi": {"wc": ["phiq1_11", "phiq1_22"]},
    "O3pQ3": {"wc": ["phiq1_33", "phiq3_33"]},
    "OpQM": {"wc": ["phiq1_33"]},
    "Opt": {"wc": ["phiu_33"]},
    "Opui": {"wc": ["phiu_11", "phiu_22"]},
    "Opdi": {"wc": ["phid_11", "phid_22", "phid_33"]},
    # Lepton Currents
    "Opl1": {"wc": ["phil1_11"]},
    "Opl2": {"wc": ["phil1_22"]},
    "Opl3": {"wc": ["phil1_33"]},
    "O3pl1": {"wc": ["phil3_11"]},
    "O3pl2": {"wc": ["phil3_22"]},
    "O3pl3": {"wc": ["phil3_33"]},
    "Ope": {"wc": ["phie_11"]},
    "Opmu": {"wc": ["phie_22"]},
    "Opta": {"wc": ["phie_33"]},
    # Yukawas
    "Otp": {"wc": ["uphi_33"]},
    "Ocp": {"wc": ["uphi_22"]},
    "Obp": {"wc": ["dphi_33"]},
    "Otap": {"wc": ["ephi_33"]},
    # 2L2H quark operators
    "O81qq": {
        "wc": ["qq1_1331", "qq1_2332", "qq1_1133", "qq1_2233", "qq3_1331", "qq3_2332"],
        "value": [1.0 / 4.0, 1.0 / 4.0, -1.0 / 6.0, -1.0 / 6.0, 1.0 / 4.0, 1.0 / 4.0],
    },
    "O11qq": {"wc": ["qq1_1133", "qq1_2233"]},
    "O83qq": {
        "wc": ["qq1_1331", "qq1_2332", "qq3_1133", "qq3_2233", "qq3_1331", "qq3_2332"],
        "value": [3.0 / 4.0, 3.0 / 4.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 4.0, -1.0 / 4.0],
    },
    "O13qq": {"wc": ["qq3_1133", "qq3_2233"]},
    "O8qt": {"wc": ["qu8_1133", "qu8_2233"]},
    "O1qt": {"wc": ["qu1_1133", "qu1_2233"]},
    "O8ut": {
        "wc": ["uu_1133", "uu_2233", "uu_1331", "uu_2332"],
        "value": [-1.0 / 6.0, -1.0 / 6.0, 1.0 / 2.0, 1.0 / 2.0],
    },
    "O1ut": {"wc": ["uu_1133", "uu_2233"]},
    "O8qu": {"wc": ["qu8_3311", "qu8_3322"]},
    "O1qu": {"wc": ["qu1_3311", "qu1_3322"]},
    "O8dt": {"wc": ["ud8_3311", "ud8_3322", "ud8_3333"]},
    "O1dt": {"wc": ["ud1_3311", "ud1_3322", "ud1_3333"]},
    "O8qd": {"wc": ["qd8_3311", "qd8_3322", "qd8_3333"]},
    "O1qd": {"wc": ["qd1_3311", "qd1_3322", "qd1_3333"]},
    # 4H quark operators
    "OQQ1": {"wc": ["qq1_3333"], "value": [1.0 / 2.0]},
    "OQQ8": {"wc": ["qq1_3333", "qq3_3333"], "value": [1.0 / 24.0, 1.0 / 8.0]},
    "OQt1": {"wc": ["qu1_3333"]},
    "OQt8": {"wc": ["qu8_3333"]},
    "Ott1": {"wc": ["uu_3333"]},
    # 4 leptons
    "Oll": {"wc": ["ll_1221"]},
    "Oll1111": {"wc": ["ll_1111"]},
    # 2 lepton 2 quark
    "OQe": {"wc": ["qe_3311"]},
    "OQl31": {"wc": ["lq1_1133", "lq3_1133"]},
    "OQlM1": {"wc": ["lq1_1133"]},
    "Otl1": {"wc": ["lu_1133"]},
    "Ote": {"wc": ["eu_1133"]},
}

# This creates a dictionary to go from Warsaw to SMEFiT.
# In particular, given a point in the coefficient space of the Warsaw basis,
# it tells you how to combine them to get the coefficients in the SMEFiT basis.
# These are basically the equations between Wilson coefficients.
# C_{SMEFiT} = sum_i coeff_i * C_{Warsaw, i}
# Note that the flavour structure is assumed to hold and therefore the values
# are inferred only from the 11 components.
inverse_wcxf_translate = {
    # Bosonic
    "OWWW": {"wc": ["W"]},
    "OpBox": {"wc": ["phiBox"]},
    "OpD": {"wc": ["phiD"]},
    "OpWB": {"wc": ["phiWB"]},
    "OpG": {"wc": ["phiG"]},
    "OpW": {"wc": ["phiW"]},
    "OpB": {"wc": ["phiB"]},
    "Op": {"wc": ["phi"]},
    # Dipoles
    "OtG": {"wc": ["uG_33"], "coeff": ["1/gs"]},
    "OtW": {"wc": ["uW_33"], "coeff": [1.0]},
    "OtZ": {"wc": ["uB_33", "uW_33"], "coeff": [-sw, cw]},
    # Quark Currents
    "O3pq": {"wc": ["phiq3_11"]},
    "OpqMi": {"wc": ["phiq1_11", "phiq3_11"], "coeff": [1.0, -1.0]},
    "O3pQ3": {"wc": ["phiq3_33"]},
    "OpQM": {"wc": ["phiq1_33", "phiq3_33"], "coeff": [1.0, -1.0]},
    "Opt": {"wc": ["phiu_33"]},
    "Opui": {"wc": ["phiu_11"]},
    "Opdi": {"wc": ["phid_11"]},
    # Lepton Currents
    "Opl1": {"wc": ["phil1_11"]},
    "Opl2": {"wc": ["phil1_22"]},
    "Opl3": {"wc": ["phil1_33"]},
    "O3pl1": {"wc": ["phil3_11"]},
    "O3pl2": {"wc": ["phil3_22"]},
    "O3pl3": {"wc": ["phil3_33"]},
    "Ope": {"wc": ["phie_11"]},
    "Opmu": {"wc": ["phie_22"]},
    "Opta": {"wc": ["phie_33"]},
    # Yukawas
    "Otp": {"wc": ["uphi_33"]},
    "Ocp": {"wc": ["uphi_22"]},
    "Obp": {"wc": ["dphi_33"]},
    "Otap": {"wc": ["ephi_33"]},
    # 2L2H quark operators
    "O81qq": {"wc": ["qq1_1331", "qq3_1331"], "coeff": [1.0, 3.0]},
    "O11qq": {
        "wc": ["qq1_1133", "qq1_1331", "qq3_1331"],
        "coeff": [1.0, 1.0 / 6.0, 1.0 / 2.0],
    },
    "O83qq": {
        "wc": ["qq1_1331", "qq3_1331"],
        "coeff": [1.0, -1.0],
    },
    "O13qq": {
        "wc": ["qq3_1133", "qq1_1331", "qq3_1331"],
        "coeff": [1.0, 1.0 / 6.0, -1.0 / 6.0],
    },
    "O8qt": {"wc": ["qu8_1133"]},
    "O1qt": {"wc": ["qu1_1133"]},
    "O8ut": {"wc": ["uu_1331"], "coeff": [2.0]},
    "O1ut": {"wc": ["uu_1133", "uu_1331"], "coeff": [1.0, 1.0 / 3.0]},
    "O8qu": {"wc": ["qu8_3311"]},
    "O1qu": {"wc": ["qu1_3311"]},
    "O8dt": {"wc": ["ud8_3311"]},
    "O1dt": {"wc": ["ud1_3311"]},
    "O8qd": {"wc": ["qd8_3311"]},
    "O1qd": {"wc": ["qd1_3311"]},
    # 4H quark operators
    "OQQ1": {"wc": ["qq1_3333", "qq3_3333"], "coeff": [2.0, -2.0 / 3.0]},
    "OQQ8": {"wc": ["qq3_3333"], "coeff": [8.0]},
    "OQt1": {"wc": ["qu1_3333"]},
    "OQt8": {"wc": ["qu8_3333"]},
    "Ott1": {"wc": ["uu_3333"]},
    # 4 leptons
    "Oll": {"wc": ["ll_1221"]},
    "Oll1111": {"wc": ["ll_1111"]},
    # 2 lepton 2 quark
    "OQe": {"wc": ["qe_3311"]},
    "OQl31": {"wc": ["lq3_1133"]},
    "OQlM1": {"wc": ["lq1_1133", "lq3_1133"], "coeff": [1.0, -1.0]},
    "Otl1": {"wc": ["lu_1133"]},
    "Ote": {"wc": ["eu_1133"]},
}
