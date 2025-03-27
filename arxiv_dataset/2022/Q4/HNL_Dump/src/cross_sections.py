"""Proton to D+ + D-"""
P_TO_DPDM_X = 1 # we only need the ratios of cross sections
"""Proton to D0 + D0bar"""
P_TO_D0D0_X = 2*P_TO_DPDM_X 
"""Proton to Ds + Dsbar"""
P_TO_DSDS_X = 0.2*(P_TO_D0D0_X + P_TO_DPDM_X)