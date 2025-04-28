from MESAreader import getSrcCol
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paths

root = paths.data / "MESA_output"
root_bin = root / "binaries/Z_0.0019"
b1 = (
    root_bin
    / "m1_18.0000_m2_15.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/"
)
b2 = (
    root_bin
    / "m1_20.0000_m2_17.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/"
)
b3 = (
    root_bin
    / "m1_38.0000_m2_30.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/"
)

bin_folders = [b1, b2, b3]

for b in bin_folders:
    bin_hfile = str(b) + "/binary_history.data"
    src, col = getSrcCol(bin_hfile)
    rl_relative_overflow_1 = src[:, col.index("rl_relative_overflow_1")]
    iRLOF = rl_relative_overflow_1 > 0
    m1 = src[iRLOF, col.index("star_1_mass")]
    m2 = src[iRLOF, col.index("star_2_mass")]
    # during first RLOF
    # m1 = donor star
    # m2 = accretor star
    beta_RLOF = np.absolute(m2[-1] - m2[0]) / np.absolute(m1[-1] - m1[0])
    print(b)
    print(beta_RLOF)
    print("----------------")
