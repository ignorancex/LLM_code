import astropy.io.ascii as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import simps
from astropy import constants as const
from astropy import units as u
import blueshift_util
import matplotlib.patheffects as pe


def get_fx(freq, spectrum):
    """Calculate 2-10 keV X-ray luminosity and Lbol, and fractional version

    Args:
        freq (array-like): frequency array
        spectrum (array-like): spectrum with same shape as freq and in Fnu units

    Returns:
        fx (float): X-ray fraction
        lx (float): 2-10 keV luminosity
        lbol (float): luminosity above 1e14 Hz. 
    """
    HEV = 4.13620e-15
    two_kev_eV_Hz = 2000.0 / HEV
    ten_kev_eV_Hz = 10000.0 / HEV

    select_Lx = (freq >= two_kev_eV_Hz) * (freq <= ten_kev_eV_Hz)
    select_bol = (freq >= 1e14)

    # integrate only over the selected ranged
    lx = np.fabs(simps(spectrum[select_Lx], x=freq[select_Lx]))
    lbol = np.fabs(simps(spectrum[select_bol], x=freq[select_bol]))
    fx = lx/lbol
    return (fx, lx, lbol)


def plot_seds(ax, sed_names=["m16_sed", "mean_qso_modified", "qsosed_m9.500_mdot0.468", "qsosed_m9.000_mdot0.150_hispin"],
              sed_folder="{}/seds/".format(blueshift_util.g_DataDir),
              labels=["Disc$+$PL", "Richards$+$~06", "QSOSED 1", "QSOSED 2"]):

    for i, f in enumerate(sed_names):
        wave, flambda = np.genfromtxt(
            "{}/{}.dat".format(sed_folder, f), unpack=True)
        nu = const.c.cgs.value / (wave) / 1e-8
        nulnu = wave * flambda
        lnu = nulnu / nu

        fx, lx, lbol = get_fx(nu, lnu)
        fcorr = 3.0276e46 / lbol

        if f == "m16_sed":
            nulnu = savgol_filter(nulnu, 35, 3)
            path_effects = [
                pe.Stroke(linewidth=6, foreground='#7f7f7f'), pe.Normal()]
        else:
            nulnu = savgol_filter(nulnu, 5, 3)
            path_effects = None

        ax.plot(nu, nulnu * fcorr, lw=3.5,
                path_effects=path_effects, label=labels[i])
        ax.loglog()

    vline_kwargs = {"ls": ":", "c": "k"}
    ax.axvline(1.93e15, ymax=0.78, ymin=0.06, **vline_kwargs)
    ax.axvline(1.1551424012378514e+16, ymax=0.67, ymin=0.13, **vline_kwargs)
    ax.axvline(1.557e+16, ymax=0.65, ymin=0.13, **vline_kwargs)
    ax.axvline(9.21e+16, ymax=0.49, ymin=0.06, **vline_kwargs)
    ax.text(1.93e15, 1.1e44, "1550\AA", ha="center", fontsize=12)
    ax.text(1.3e16, 1.7e44,
            r"C\textsc{iii},C\textsc{iv}", ha="center", fontsize=12)
    ax.text(9.21e+16, 1.1e44, r"C\textsc{iv} inner", ha="center", fontsize=12)

    ax.set_xlim(1.1e14, 1e19)
    ax.set_ylim(1e44, 4e46)
    ax.set_ylabel(r"$\nu L_\nu~({\rm erg~s^{-1}})$", fontsize=18, labelpad=-1)
    ax.set_xlabel(r"$\nu~({\rm Hz})$", fontsize=18, labelpad=-1)
    ax.legend(frameon=False, labelspacing=0.25,
              borderpad=0.25, loc="upper right")


def make_figure():
    print("Making figure 12...", end="")
    blueshift_util.set_cmap_cycler("magma_r", 6)
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10, 4))
    gs1 = GridSpec(1, 2, right=0.58, left=0.065,
                   bottom=0.15, top=0.98, wspace=0.05)
    gs2 = GridSpec(1, 1, left=0.66, right=0.99, bottom=0.15, top=0.98)


    data_dir = "{}/sed_change".format(blueshift_util.g_DataDir)
    #data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..', 'Data/specs_f840486/'))

    plt.figure()
    iplot = 0

    fname_list = [["run47.spec", "run47_meanqso.spec", "run47_qsosed1.spec", "run47_qsosed2.spec", "run47_aniso.spec"],
                  ["run128.spec", "run128_meanqso.spec", "run128_qsosed1.spec", "run128_qsosed2.spec", "run128_aniso.spec"]]

    labels = ["Disc$+$PL", "Richards$+$~06",
              "QSOSED 1", "QSOSED 2", "Anisotropic Disc"]
    mod_name = ["e)", "n)"]

    for i, fnames in enumerate(fname_list):
        # plt.subplot(1,3,i+1)
        ax = fig.add_subplot(gs1[0, i])
        for j, f in enumerate(fnames):

            # try:
            s = io.read("{}/{}".format(data_dir, f))
            # except astropy.io.ascii.core.InconsistentTableError:
            #     print ("Error for", f)

            wavelength = s["Lambda"]
            velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs
            velocity = velocity.to(u.km/u.s)

            fmax = 0

            # for i in range(len(angles)):
            angle = blueshift_util.fiducial_angle
            colname = "A{:02d}P0.50".format(int(angle))

            fcont = blueshift_util.fit_continuum(wavelength, s[colname])
            #flux = savgol_filter(s[colname]/fcont, 15, 3)
            flux = savgol_filter(s[colname], 15, 3)

            ls = "-"
            color = "C{}".format(j)
            zord = 2
            lw = 3
            label_use = None
            if f == "run47.spec" or f == "run128.spec":
                path_effects = [
                    pe.Stroke(linewidth=6, foreground='#7f7f7f'), pe.Normal()]
            elif f == "run47_aniso.spec" or f == "run128_aniso.spec":
                path_effects = [
                    pe.Stroke(linewidth=3, foreground='#7f7f7f'), pe.Normal()]
                ls = "--"
                color = "C0"
                zord = 1
                lw = 1.5
                label_use = labels[j]
            else:
                path_effects = None
            ax.plot(velocity/1e3, flux/4, lw=lw, path_effects=path_effects,
                    label=label_use, c=color, ls=ls, zorder=zord)
        if i == 0:
            ax.legend(frameon=True)
        ax.set_xlim(-8, 8)
        ax.set_ylim(1, 20)

        ax.axvline(0, ls="--", alpha=0.5, color="k")
        ax.set_yscale("log")
        ax.text(-6, 10, mod_name[i], fontsize=18)
        if i > 0:
            # if iplot == 12:
            ax.set_yticklabels([])
        else:
            #ax.legend(frameon=True, labelspacing=0.25, borderpad=0.25, loc="upper right")
            ax.set_ylabel(r"$F_\lambda$~(Arb.)", fontsize=18, labelpad=-1)

        ax.set_xlabel(
            "Velocity~($10^3$~km~s$^{-1}$)", fontsize=18, labelpad=-1)

        iplot += 1

    ax = fig.add_subplot(gs2[0, 0])
    plot_seds(ax, labels=labels)

    blueshift_util.save_paper_figure("fig12.pdf", fig=fig)
    print("Done.")


if __name__ == "__main__":
    blueshift_util.set_plot_defaults()
    make_figure()
