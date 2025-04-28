from lib_engineered import three_panel_plot_s_h_he, sorter_engineered_profiles
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paths
import warnings

if __name__ == "__main__":
    root = paths.data / "MESA_output/"
    nonrot18 = root / "single_stars/Z_0.0019/18_rot0.0/"
    nonrot20 = root / "single_stars/Z_0.0019/20_rot0.0/"
    nonrot36 = root / "single_stars/Z_0.0019/36_rot0.0/"

    root_TAMS = root / "engineered_stars/TAMS_models/"
    init_model18 = root_TAMS / "18_rot0_to_TAMS/LOGS/TAMS.data"
    init_model20 = root_TAMS / "20_rot0_to_TAMS/LOGS/TAMS.data"
    init_model36 = root_TAMS / "36_rot0_to_TAMS/LOGS/TAMS.data"

    root_eng = str(root / "engineered_stars/same_core/")
    grid_folders18 = sorted(
        glob.glob(root_eng + "/grid18/*.*/LOGS/"), key=sorter_engineered_profiles
    )
    grid_folders20 = sorted(
        glob.glob(root_eng + "/grid20/*.*/LOGS/"), key=sorter_engineered_profiles
    )
    grid_folders36 = sorted(
        glob.glob(root_eng + "/grid36/*.*/LOGS/"), key=sorter_engineered_profiles
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

    # make TAMS_profiles plot
    grid_folders1 = grid_folders18
    init_model1 = str(init_model18)
    accretor1 = str(accretor18)
    nonrot1 = str(nonrot18)

    grid_folders2 = grid_folders20
    init_model2 = str(init_model20)
    accretor2 = str(accretor20)
    nonrot2 = str(nonrot20)

    grid_folders3 = grid_folders36
    init_model3 = str(init_model36)
    accretor3 = str(accretor30)
    nonrot3 = str(nonrot36)

    fig_name = paths.figures / "TAMS_profiles.pdf"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
