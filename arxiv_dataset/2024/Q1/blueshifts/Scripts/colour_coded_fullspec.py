import matplotlib.pyplot as p
from astropy import units as u
import numpy as np
import py_plot_util as util
import os
from plot_norm import *
from scipy.signal import savgol_filter
import jm_util
import blueshift_util
import os
import py_read_output as rd
import astropy.io.ascii as io

jm_util.set_mod_defaults()
jm_util.set_times()
W0 = 1700
Alpha, AlphaString, data_dir, speclist = blueshift_util.get_filenames_for_alpha(
    sys.argv)


def plot_full_spectra(ax, files, colors, args, folder):

    for i in args:
        wavelength, fl = blueshift_util.read_fits_spectrum(folder+files[i])
        select = (wavelength > 1280) * (wavelength < 1999)
        wavelength = wavelength[select]
        fl = fl[select]
        f0 = blueshift_util.get_flux_at_wavelength(wavelength, fl, W0)
        fl = savgol_filter(fl/f0, 15, 3)
        #gradient_fill(wavelength, fl, fill_color=colors[i], ax=ax, lw=2.3)
        ax.plot(wavelength, fl, c=colors[i], lw=3, alpha=1)

    delta = (1e9 / 3e10) * 1550
    ax.vlines([1550], 3.3, 3.7, color="k")
    #ax.fill_between(np.linspace(1550 - delta, 1550 + delta, 10), y1=0.55, y2 = 4.5, alpha=0.25, color="k")

    ax.set_yscale("log")
    # ax.set_xlim(1280,1999)
    ax.set_ylim(0.55, 4.5)
    ax.set_ylabel(r"$F_\lambda$~(Arb.)", fontsize=18, labelpad=-8)
    ax.set_xlabel(r"Rest Wavelength (\AA)", fontsize=18, labelpad=-1)

    ax.set_yticks([0.6, 1, 2, 3, 4])
    ax.set_yticklabels([str(i) for i in [0.6, 1, 2, 3, 4]])
    lines = [1305, 1335, 1398, 1640, 1666, 1857, 1909]
    # "C \textsc{ivi}"
    labels = [r"O\textsc{i}", r"C\textsc{ii}", r"Si\textsc{iv}$+$O\textsc{iv}]",
              r"He \textsc{ii}", r"O \textsc{iii}]", r"Al \textsc{iii}", r"C \textsc{iii}]"]
    y = [0.58, 0.77, 0.58, 0.58, 0.77, 0.77, 0.58]
    for i, l in enumerate(lines):
        ax.text(l - 10, y[i], labels[i], fontsize=16,
                horizontalalignment="left")

    ax.vlines([lines], 0.65, 0.75, color="k")
    ax.text(1550, 3.8, r"C\textsc{iv}~1550\AA",
            fontsize=16, color="k", horizontalalignment="center")


speclist = [f for f in os.listdir(data_dir)]
cmap_names = blueshift_util.cmap_dict
for itb, tb in enumerate(["thmin45"]):
    print(itb)

    norm, mappable = blueshift_util.get_norm_mappable(cmap=cmap_names[tb])

    #fnames = [f for f in os.listdir(data_dir) if ".spec" in f[-6:] and tb in f]
    # if
    fnames = [f for f in os.listdir(data_dir) if ".spec" in f[-6:] and (
        (tb in f and "thmin45_rv1e19_vinf1p0_mdotw5p0" in f) or "run128_aniso" in f) and "tau" not in f]
    print(fnames, data_dir)
    for f in fnames:
        s1 = rd.read_spectrum(data_dir + f)

        LW = 3

        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(122)

        pf_root = f[:-5]
        pf = rd.read_pf("{}/{}".format(data_dir, pf_root))
        theta1 = float(pf["SV.thetamin(deg)"])
        theta2 = float(pf["SV.thetamax(deg)"])
        theta_b = 0.5 * (theta1 + theta2)
        theta1 = 80
        print(theta2)

        angles = np.arange(5, theta2+3, 5)

        for iangle, angle in enumerate(angles):
            colname = "A{:02d}P0.50".format(int(angles[iangle]))
            fl = s1[colname]
            #f = s1["Disk"]
            w = s1["Lambda"]
            f0 = blueshift_util.get_flux_at_wavelength(w, fl, W0)
            #cc = blueshift_util.get_continuum(w, f/f0, [1215, 1240, 1550, 1640, 1400, 1000, 1800,1909,2800], lmin=800, lmax = 3000, deg = 4)
            deltaf = ((1 * (len(angles))) - (1 * (iangle + 1)))
            fl = savgol_filter(fl/f0, 15, 3) + deltaf
            plt.plot(w, fl, alpha=1, c=mappable.to_rgba(
                angles[iangle]), zorder=5, lw=2.5)
            plt.text(1750, 1.07 + deltaf, "${:.0f}^\circ$".format(angle),
                     ha="center", c=mappable.to_rgba(angles[iangle]), fontsize=12)
        plt.xlim(1150, 2000)
        plt.ylim(0, 15)
        plt.axvline(1550, ls="--", color="k")
        plt.xlabel(r"Wavelength~(\AA)", fontsize=18)
        plt.ylabel(r"$F_\lambda/F_{1700}+{\rm offset}$", fontsize=18)

        ax = plt.subplot(121)
        d = io.read(data_dir+f[:-5] + ".master.txt")
        x, z, c4, _ = blueshift_util.wind_to_masked(
            d, "c4", return_inwind=True, ignore_partial=True)
        x, z, ne, _ = blueshift_util.wind_to_masked(
            d, "ne", return_inwind=True, ignore_partial=True)
        x, z, vz, _ = blueshift_util.wind_to_masked(
            d, "v_z", return_inwind=True, ignore_partial=True)
        cbar_mappable = ax.pcolormesh(
            x/1e18, z/1e18, np.log10(c4), vmax=0, vmin=-4, cmap="viridis")
        print(np.max(c4))
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)

        plt.xlabel(r"$x~(10^{18}~{\rm cm})$")
        plt.ylabel(r"$z~(10^{18}~{\rm cm})$")
        cax = plt.gcf().add_axes([0.3, 0.28, 0.12, 0.04])
        cbar = plt.colorbar(mappable=cbar_mappable, cax=cax,
                            orientation="horizontal", extend="min")
        cbar.set_label(
            r"$\log[{\rm C}~\textsc{iv}~{\rm fraction}]$", labelpad=-2)
        # cbar.set_clim(-4,0)
       # plt.loglog()

        xx = np.linspace(0, 1e20, 1000)
        for iangle, angle in enumerate(angles):
            ax.plot(xx, xx*np.tan(np.radians(90-angle)),
                    c=mappable.to_rgba(angles[iangle]), alpha=1, ls="-", lw=1.5)

        savename = "OtherFigures/cc_spec_{}.pdf".format(f[:-5])
        print(savename)
        #plt.subplots_adjust(left=0.07, right=0.98,top=0.98, bottom=0.14)
        plt.tight_layout(pad=0.05, w_pad=0.05)
        plt.savefig(savename, dpi=300)
