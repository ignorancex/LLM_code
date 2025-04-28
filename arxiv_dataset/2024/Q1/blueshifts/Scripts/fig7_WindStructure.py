import astropy.io.ascii as io
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import blueshift_util


def make_figure():
    print("Making figure 7...", end="")

    Alpha, AlphaString, data_dir, speclist = blueshift_util.get_filenames_for_alpha(
        "-1")

    cmap = "Spectral_r"

    plt.figure()
    iplot = 0

    plt.figure(figsize=(12, 7))

    fnames = [f for f in speclist]

    rho2nh, _ = blueshift_util.rho2nh()

    v90_all = np.zeros(len(fnames))
    ia1 = 0
    ia2 = 9
    for f in fnames:
        full_fname = "{}/{}".format(data_dir, f[:-5])
        #print (full_fname)
        #os.system("windsave2table86g {}".format(full_fname))
        d = io.read(full_fname + ".master.txt")

        #util.run_py_wind(full_fname, cmds=["1", "L", "0", "3", "s", "q"], vers="86g")
        _, _, lc4 = blueshift_util.read_pywind(
            "{}.lineC4.dat".format(full_fname))
        _, _, vol = blueshift_util.read_pywind("{}.vol.dat".format(full_fname))

        plt.subplot(3, 6, iplot+1)
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

        mu = 1.41
        #print (rho2nh)
        ne = rho * rho2nh
        mappable = plt.scatter(np.log10(ne), np.log10(
            vz/1e5), c=np.log10(lc4), s=5, vmin=37.5, vmax=42.5, cmap=cmap)

        vz_flt = vz.flatten()
        lc4_flt = lc4.flatten()
        lcf_cdf = np.cumsum(lc4_flt) / np.sum(lc4_flt)
        v90 = vz_flt[np.argmin(np.fabs(lcf_cdf-0.9))]/1e5
        v90_all[iplot] = v90

        plt.grid(ls=":")
        if iplot not in [0, 6, 12]:
            plt.gca().set_yticklabels([])
        if iplot < 12:
            plt.gca().set_xticklabels([])
        if iplot == 12:
            plt.xlabel(r"$\log[n_H~({\rm cm}^{-3})]$", fontsize=20)
            plt.ylabel(r"$\log[v_z~({\rm km~s}^{-1})]$", fontsize=20)
        if iplot == 2 and AlphaString == "combo":
            plt.text(
                10.86, 4.6, r"$\leftarrow \alpha = 0.5 | \alpha = 1 \rightarrow$", ha="center")

        if iplot in [0, 1, 2, 6, 7, 8, 12, 13, 14]:
            plt.text(6, 2.5, string.ascii_lowercase[ia1]+")")
            ia1 += 1
        else:
            plt.text(6, 2.5, string.ascii_lowercase[ia2]+")")
            ia2 += 1

        if "run47" in f or "run128" in f:
            plt.scatter(9, 4, marker="*", ec="k", fc="None", s=140)

        iplot += 1
        plt.xlim(5.2, 11)
        plt.ylim(1.9, 4.5)

    ax = plt.gcf().add_axes([0.62, 0.05, 0.2, 0.03])
    cbar = plt.colorbar(mappable=mappable, cax=ax,
                        orientation="horizontal", extend="both")
    plt.text(1.07, 0.1, r"log[$L_{1550}$~(erg~s$^{-1}$)]",
             transform=ax.transAxes, fontsize=16)
    plt.subplots_adjust(top=0.96, left=0.06, bottom=0.14,
                        right=0.98, wspace=0.05, hspace=0.05)
    figure_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'Figures'))

    blueshift_util.save_paper_figure("fig7.pdf")
    #print (np.mean(v90_all), np.median(v90_all))
    print("Done.")


if __name__ == "__main__":
    blueshift_util.set_plot_defaults()
    make_figure()
