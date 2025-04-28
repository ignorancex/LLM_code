import astropy.io.ascii as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from astropy import constants as const
from astropy import units as u
import os
import blueshift_util
from tqdm import tqdm


def make_figure():
    print("Making figure 6...")

    data_dir = blueshift_util.g_DataDir
    cmap_names = blueshift_util.cmap_dict
    th_label = [70, 45, 20]

    all_files = []
    all_roots = []
    for alpha in ["0.5", "1", "1.5"]:
        alpha_dir = "{}/alpha{}".format(data_dir, alpha)
        for f in os.listdir(alpha_dir):
            if ".spec" in f and "thmin" in f:
                full_fname = "{}/{}".format(alpha_dir, f)
                all_files.append(full_fname)
                all_roots.append(f)

    plt.figure(figsize=(6, 5))

    for itb, tb in enumerate(["thmin70", "thmin45", "thmin20"]):
        fnames = [f for f in all_files if tb in f]
        roots = [f for f in all_roots if tb in f]

        _, mappable = blueshift_util.get_norm_mappable(cmap=cmap_names[tb])
        ilabel = 0

        print("Computing blueshifts and skews for all models with {}".format(tb))

        for j, f in enumerate(tqdm(fnames)):

            try:
                s = io.read(f)
            except io.ascii.core.InconsistentTableError:
                print("Error for", f)

            wavelength = s["Lambda"]
            velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs
            velocity = velocity.to(u.km/u.s)

            nn = int(roots[j][3:6].strip("_"))

            pf_root = f[:-5]
            pf = blueshift_util.read_pf(pf_root)
            theta1 = float(pf["SV.thetamin(deg)"])

            angles = np.arange(5, int(theta1)+5, 5)

            blueshifts = np.zeros_like(angles, dtype=float)
            skews = np.zeros_like(angles, dtype=float)
            ews = np.zeros_like(angles, dtype=float)

            for i in range(len(angles)):
                colname = "A{:02d}P0.50".format(angles[i])

                fcont = blueshift_util.fit_continuum(wavelength, s[colname])
                try:
                    flux = savgol_filter(s[colname]/fcont, 5, 3)
                except:
                    flux = s[colname]/fcont
                # get blueshift
                blueshifts[i] = blueshift_util.get_blueshift(
                    velocity/(u.km / u.s), s[colname], fcont)
                vpeak, fpeak, skew = blueshift_util.get_skew(
                    velocity/(u.km / u.s), flux, blueshifts[i])

                skews[i] = (-vpeak)-blueshifts[i]
                ews[i] = blueshift_util.get_ew(
                    wavelength, velocity/(u.km / u.s), s[colname], fcont)

            if ilabel == 0:
                label = r"$\theta_{{\rm min}} = {:.0f}^\circ$".format(
                    th_label[itb])
                ilabel += 1
            else:
                label = None
            EW_CRIT = 20
            plot_kwargs = {"facecolors": mappable.to_rgba(
                40), "edgecolors":  "None", "s": 70, "alpha": 0.8}
            plt.scatter(1e50*blueshifts[ews > EW_CRIT], skews[ews >
                        EW_CRIT], zorder=2, label=label, **plot_kwargs)
            if nn > 81:
                plt.scatter(blueshifts[ews > EW_CRIT], skews[ews >
                            EW_CRIT], zorder=2, **plot_kwargs, marker="^")
            else:
                plt.scatter(blueshifts[ews > EW_CRIT],
                            skews[ews > EW_CRIT], zorder=2, **plot_kwargs)

    print("Reading observational data and finishing plot...")

    blueshifts, skew = np.genfromtxt(
        "{}/blueshifts_skews_composites.dat".format(data_dir), unpack=True)
    #blueshifts, ews, skew, vpeak = blueshift_util.measure_blueshifts()
    plt.scatter(blueshifts, skew, c="k", zorder=3,
                marker="x", label="Composites")

    plt.legend(frameon=False, loc=2, handletextpad=0.1,
               fontsize=18, borderaxespad=0.1, borderpad=0.1)
    plt.xlabel(r"C~\textsc{iv}~Blueshift~(km~s$^{-1}$)", fontsize=18)
    plt.ylabel(r"C~\textsc{iv}~Skew~(km~s$^{-1}$)", fontsize=18, labelpad=-3)
    plt.xlim(-2000, 2999)
    plt.ylim(-4000, 4000)
    plt.grid(ls=":")
    blueshift_util.save_paper_figure("fig6.pdf")
    print("Done.")


if __name__ == "__main__":
    blueshift_util.set_plot_defaults()
    make_figure()
