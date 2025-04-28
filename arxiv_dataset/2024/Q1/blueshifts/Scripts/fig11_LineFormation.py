import astropy.io.ascii as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from astropy import constants as const
from astropy import units as u
import blueshift_util
import matplotlib.patheffects as pe


def vescape(R, M=1e9):
    vesc = np.sqrt(2.0 * M * const.GM_sun.cgs.value / R)
    return vesc


def vkep(R, M=1e9):
    vk = np.sqrt(M * const.GM_sun.cgs.value / R)
    return vk


def vel_law(l, R_v=1e19, r0=3e16, v0=1e6, fesc=1.0, alpha=1.0, theta=45.0):
    vinf = fesc * vescape(r0)
    l_over_rv_to_alpha = (l / R_v)**alpha
    v = v0 + (vinf - v0) * (l_over_rv_to_alpha / (1.0 + l_over_rv_to_alpha))
    x = r0 + (l * np.cos(np.radians(theta)))
    vrot = vkep(r0, M=1e9) * r0 / x

    return (v, vinf, vrot)


def make_figure():
    print("Making figure 11...", end="")

    data_dir = "{}/steppar_specs_alpha1/".format(blueshift_util.g_DataDir)

    plt.figure()
    angles_all = np.arange(5, 90, 5)
    iplot = 0

    rho2nh, _ = blueshift_util.rho2nh()

    root = "run47"

    fnames = ["_alpha0.5.spec", "_alpha0.75.spec",
              "_alpha1.spec", "_alpha1.25.spec", "_alpha1.5.spec"]

    blueshift_util.set_cmap_cycler("viridis_r", 5)

    plt.figure(figsize=(13, 6))
    alphas = [0.5, 0.75, 1, 1.25, 1.5]
    for f in fnames:
        full_fname = "{}/{}{}".format(data_dir, root, f[:-5])
        d = io.read(full_fname + ".master.txt")
        alpha = alphas[iplot]

        plt.subplot(2, 7, iplot+1)

        color = "C" + str(iplot)
        s = io.read("{}/{}{}".format(data_dir, root, f))
        wavelength = s["Lambda"]
        velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs
        velocity = velocity.to(u.km/u.s)
        angle = blueshift_util.fiducial_angle
        colname = "A{:02d}P0.50".format(int(angle))

        fcont = blueshift_util.fit_continuum(wavelength, s[colname])
        flux = savgol_filter(s[colname]/fcont, 15, 3)

        # if f == "_alpha1.spec":
        path_effects = [pe.Stroke(linewidth=6, foreground='k'), pe.Normal()]
        # else:
        #    path_effects = None
        plt.plot(velocity/1e3, flux, lw=3, path_effects=path_effects, c=color)
        plt.xlim(-8, 8)
        plt.ylim(0.9, 4.5)
        plt.text(-2, 4.1, r"$\alpha = {}$".format(alpha))

        #util.run_py_wind(full_fname, cmds=["1", "L", "0", "3", "s", "q"])
        _, _, lc4 = blueshift_util.read_pywind(
            "{}.lineC4.dat".format(full_fname))
        _, _, vol = blueshift_util.read_pywind("{}.vol.dat".format(full_fname))

        if iplot == 0:
            plt.ylabel(r"Normalised $F_\lambda$", fontsize=18)
            plt.xlabel(r"Velocity~$({\rm km~s}^{-1})]$",
                       fontsize=18, labelpad=-2)
        else:
            plt.gca().set_yticklabels([])

        plt.subplot(2, 7, 7+iplot+1)
        x, z, rho, _ = blueshift_util.wind_to_masked(
            d, "rho", return_inwind=True, ignore_partial=True)
        x, z, c4, _ = blueshift_util.wind_to_masked(
            d, "c4", return_inwind=True, ignore_partial=True)
        x, z, ne, _ = blueshift_util.wind_to_masked(
            d, "ne", return_inwind=True, ignore_partial=True)
        x, z, vz, _ = blueshift_util.wind_to_masked(
            d, "v_z", return_inwind=True, ignore_partial=True)
        x, z, vx, _ = blueshift_util.wind_to_masked(
            d, "v_x", return_inwind=True, ignore_partial=True)
        x, z, vy, _ = blueshift_util.wind_to_masked(
            d, "v_y", return_inwind=True, ignore_partial=True)
        vz = np.sqrt(vz * vz + vx * vx)

        ne = rho * rho2nh
        mappable = plt.scatter(np.log10(vy/1e5), np.log10(vz/1e5),
                               c=np.log10(lc4), s=10, vmin=37.5,
                               vmax=42.5, cmap="Spectral_r")
        vv = np.linspace(1, 6, 1000)
        plt.plot(vv, vv, ls="--", c="k")
        if iplot == 0:
            plt.xlabel(r"$\log[v_{\phi}~({\rm km~s}^{-1})]$",
                       fontsize=18, labelpad=-2)
            plt.ylabel(r"$\log[v_{l}~({\rm km~s}^{-1})]$", fontsize=18)
        else:
            plt.gca().set_yticklabels([])
        plt.grid(ls=":")
        plt.xlim(1.3, 4.3)
        plt.ylim(1.3, 4.3)

        iplot += 1

    ax_cbar = plt.subplot2grid((20, 21), (17, 16), colspan=4, rowspan=1)
    cbar = plt.colorbar(cax=ax_cbar, shrink=0.5, fraction=0.1,
                        mappable=mappable, orientation="horizontal", extend="both")
    cbar.set_label(r"log[$L_{1550}$~(erg~s$^{-1}$)]")
    ax2 = plt.subplot2grid((10, 21), (1, 16), colspan=5, rowspan=6)

    l = np.logspace(14, 20, 1000)

    for alpha in alphas:
        R_v = 1e19
        path_effects = [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
        v, vinf, vrot = vel_law(l, R_v=R_v, r0=3e16,
                                v0=1e6, fesc=1.0, alpha=alpha, theta=45.0)
        ax2.plot(l/R_v, v/vinf, path_effects=path_effects, lw=3,
                 label=r"$v_l/v_\infty, \alpha={}$".format(alpha))

    ax2.plot(l/R_v, vrot/vinf, ls="--", c="k", label=r"$v_\phi/v_\infty$")
    plt.semilogx()
    ax2.set_xlim(1e-3, 10)
    ax2.set_xlabel("$l/R_v$", fontsize=16)
    ax2.legend(frameon=False, loc="upper left", labelspacing=0.1,
               borderaxespad=0.4, handletextpad=0.4)
    plt.title("Velocity along streamlines")
    plt.subplots_adjust(wspace=0.05, bottom=0.1, hspace=0.25,
                        left=0.06, right=0.985, top=0.98)
    blueshift_util.save_paper_figure("fig11.pdf")
    print("Done.")


if __name__ == "__main__":
    blueshift_util.set_plot_defaults()
    make_figure()
