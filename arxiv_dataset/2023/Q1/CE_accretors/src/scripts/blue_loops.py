from lib_plot_bin import getSrcCol
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

fig = plt.figure()
gs = gridspec.GridSpec(110, 120)
gs.update(wspace=0, hspace=0)  # top=1.1)
ax = plt.subplot(gs[:, :])

# get default colors
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

for f in bin_folders:
    c = colors[bin_folders.index(f)]
    LOGS2 = f / "LOGS2/"
    input_file = str(LOGS2) + "/history.data"
    src, col = getSrcCol(input_file, True, True)
    # T = src[:, col.index("log_Teff")]
    # L = src[:, col.index("log_L")]
    # ax.plot(T, L)
    # # find blue loop
    # center_h1 = src[:, col.index("center_h1")]
    center_he4 = src[:, col.index("center_he4")]
    R = src[:, col.index("radius")]
    t = src[:, col.index("star_age")]
    ax.scatter(t, R, c=center_he4, cmap=plt.cm.seismic)

plt.show()
