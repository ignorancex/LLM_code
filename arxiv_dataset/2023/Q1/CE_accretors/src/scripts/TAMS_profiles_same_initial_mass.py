from lib_plot_bin import plot_BE_r
from lib_engineered import (
    get_M_boundary,
    plot_XY,
    get_M_boundary,
    plot_entropy,
    sorter_engineered_profiles,
    plot_s_h_he,
    three_panel_plot_s_h_he,
)
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import paths

root = paths.data / "MESA_output/"
nonrot15 = root / "single_stars/Z_0.0019/15_rot0.0/"
nonrot17 = root / "single_stars/Z_0.0019/17_rot0.0/"
nonrot30 = root / "single_stars/Z_0.0019/30_rot0.0/"

root_TAMS = root / "engineered_stars/TAMS_models/"
init_model15 = root_TAMS / "15_rot0_to_TAMS/LOGS/TAMS.data"
init_model17 = root_TAMS / "17_rot0_to_TAMS/LOGS/TAMS.data"
init_model30 = root_TAMS / "30_rot0_to_TAMS/LOGS/TAMS.data"

root_eng = str(root / "engineered_stars/same_core/")
grid_folders15 = sorted(
    glob.glob(root_eng + "/grid15/*.*/LOGS/"), key=sorter_engineered_profiles
)
grid_folders17 = sorted(
    glob.glob(root_eng + "/grid17/*.*/LOGS/"), key=sorter_engineered_profiles
)
grid_folders30 = sorted(
    glob.glob(root_eng + "/grid30/*.*/LOGS/"), key=sorter_engineered_profiles
)

root_accretors = root / "binaries/Z_0.0019"
accretor18 = (
    root_accretors
    / "m1_18.0000_m2_15.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
)
accretor20 = (
    root_accretors
    / "m1_20.0000_m2_17.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
)
accretor30 = (
    root_accretors
    / "m1_38.0000_m2_30.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
)


# make same initial mass TAMS_profile
grid_folders1 = grid_folders15
init_model1 = str(init_model15)
accretor1 = str(accretor18)
nonrot1 = str(nonrot15)

grid_folders2 = grid_folders17
init_model2 = str(init_model17)
accretor2 = str(accretor20)
nonrot2 = str(nonrot17)

grid_folders3 = grid_folders30
init_model3 = str(init_model30)
accretor3 = str(accretor30)
nonrot3 = str(nonrot30)

fig_name = paths.figures / "TAMS_profiles_same_initial_mass.pdf"

three_panel_plot_s_h_he(
    grid_folders1,
    init_model1,
    accretor1,
    nonrot1,
    grid_folders2,
    init_model2,
    accretor2,
    nonrot2,
    grid_folders3,
    init_model3,
    accretor3,
    nonrot3,
    fig_name=fig_name,
)
