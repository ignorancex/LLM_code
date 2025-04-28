from MESAreader import getSrcCol, Rsun_cm, secyer
import paths
import glob


def get_t_conv_core_from_pfile(pfile):
    src, col = getSrcCol(str(pfile))
    # flip arrays to go from center outwards
    mix_type = src[::-1, col.index("mixing_type")]
    conv_vel = src[::-1, col.index("conv_vel")]  # cm/s
    r = src[::-1, col.index("radius")] * Rsun_cm  # cm
    t_conv = 0  # sec
    i = 0
    while mix_type[i] == 1:  # check only core convective zone
        t_conv += r[i] / conv_vel[i]
        i += 1
    return t_conv / secyer  # yr


if __name__ == "__main__":
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

    TAMS_folders = paths.data / "MESA_output/engineered_stars/TAMS_models/"
    print(str(TAMS_folders) + "*/LOGS/")
    for b in glob.glob(str(TAMS_folders) + "/*/LOGS/"):
        print(b)
        TAMSdata = glob.glob(str(b) + "/*TAMS*.data")
        # if TAMSdata == []:
        #     LOGS2 = b / "LOGS2"
        #     TAMSdata = glob.glob(str(LOGS2) + "/*TAMS*.data")
        for d in TAMSdata:
            print(d)
            t_conv = get_t_conv_core_from_pfile(d)
            print(b, t_conv)
