import numpy as np
from MESAreader import getSrcCol, Rsun_cm


def get_Ertl_param(pfile):
    """ Explodability parameters from Ertl et al. 2016"""
    src, col = getSrcCol(pfile)
    s = src[:, col.index("entropy")]
    m = src[:, col.index("mass")]
    r = src[:, col.index("radius")]
    i = np.argmin(np.absolute(s-4.0))
    dm = 0.3
    j = np.argmin(np.absolute((m[i]+dm)-m))
    M4 = m[i]
    mu4 = (m[j]-m[i])/((r[j]-r[i])*Rsun_cm/1e8) # Eq. 5 in Ertl+2016
    return M4, mu4


if __name__ == "__main__":
    root = "../MESA_results/"

    small = root+'40_rot0.6_small_net/'
    LOGS_small = small+"LOGS1/"
    hfile_small = LOGS_small+'history.data'
    pfile_small = LOGS_small+'CHE_single_core_collapse.data'
    src_small, col_small = getSrcCol(hfile_small)

    large = root+'40_rot0.6_large_net/'
    LOGS_large = large+'LOGS1/'
    hfile_large = LOGS_large+'history.data'
    pfile_large = LOGS_large+'CHE_single_core_collapse.data'
    src_large, col_large = getSrcCol(hfile_large)

    # compactness
    xi_small = src_small[-1, col_small.index('compactness_parameter')]
    print("xi2.5 small",
          f"{xi_small:.3f}")
    xi_large = src_large[-1, col_large.index('compactness_parameter')]
    print("xi2.5 large",
          f"{xi_large:.3f}")
    print("relative change in xi", f"{((xi_small - xi_large)/xi_large)*100:.0f}", "%")

    # Fe cores
    Fe_core_small = src_small[-1, col_small.index("fe_core_mass")]
    print("fe core small",
          f"{Fe_core_small:.2f}")
    Fe_core_large = src_large[-1, col_large.index("fe_core_mass")]
    print("fe core large",
          f"{Fe_core_large:.2f}")

    # Ertl parameters
    M4_small, mu4_small = get_Ertl_param(pfile_small)
    print("small:", "M4=", f"{M4_small:.2f}", "mu4", f"{mu4_small:.2f}")
    M4_large, mu4_large = get_Ertl_param(pfile_large)
    print("large:", "M4=", f"{M4_large:.2f}", "mu4", f"{mu4_large:.2f}")

    # Ye fractional change
    src_pfile_small, col_pfile_small = getSrcCol(pfile_small)
    ye_small = src_pfile_small[-1, col_pfile_small.index("ye")]
    src_pfile_large, col_pfile_large = getSrcCol(pfile_large)
    ye_large = src_pfile_large[-1, col_pfile_large.index("ye")]
    print("% diff in central Ye:", f"{((ye_small - ye_large)/ye_large)*100:.0f}")
