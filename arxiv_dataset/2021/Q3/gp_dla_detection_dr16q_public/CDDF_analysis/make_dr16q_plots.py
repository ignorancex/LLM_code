"""
Make plots for the Multi-DLA paper
"""
import os
import pathlib
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from astropy.io import fits

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from .qso_loader import QSOLoader, file_loader
from .qso_loader_dr16q import QSOLoaderDR16Q
from .set_parameters import *
from .dla_data import dla_data

from .effective_optical_depth import effective_optical_depth

cmap = cm.get_cmap("viridis")

# change fontsize
matplotlib.rcParams.update(
    {"font.size": 12, "axes.labelsize": 16, "axes.titlesize": 14}
)

# matplotlib.use('PDF')
save_figure = lambda filename: plt.savefig(
    "{}.pdf".format(filename), format="pdf", dpi=300
)


def do_procedure_plots(qsos_full_int: QSOLoader, qsos_original: QSOLoader):
    # scaling factor between rest_wavelengths to pixels
    min_lambda = qsos_full_int.GP.min_lambda - 10
    max_lambda = qsos_full_int.GP.max_lambda + 10
    scale = 1 / (max_lambda - min_lambda)

    nv_wavelength = 1240.81
    oi_wavelength = 1305.53
    cii_wavelength = 1335.31
    siv_wavelength = 1399.8

    # compare different learned mus
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(
        qsos_full_int.GP.rest_wavelengths,
        qsos_full_int.GP.mu,
        label=r"re-trained $\mu$",
    )
    ax.plot(
        qsos_original.GP.rest_wavelengths,
        qsos_original.GP.mu,
        label=r"Ho-Bird-Garnett (2020) $\mu$",
        color="lightblue",
    )

    ax.legend()
    ax.set_xlabel(r"Rest-frame Wavelength $(\AA)$")
    ax.set_ylabel(r"Normalized Flux")
    ax.set_xlim([min_lambda, max_lambda])

    ax02 = ax.twiny()
    ax02.set_xticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ]
    )
    ax02.set_xticklabels(
        [r"Ly $\infty$", r"Ly $\beta$", r"Ly $\alpha$", r"NV", r"OI", r"CII", r"SIV"]
    )
    plt.tight_layout()
    save_figure("mu_changes")
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    # compare different learned omegas
    ax.plot(
        qsos_full_int.GP.rest_wavelengths,
        np.exp(qsos_full_int.GP.log_omega),
        label=r"re-trained $\omega$",
    )
    ax.plot(
        qsos_original.GP.rest_wavelengths,
        np.exp(qsos_original.GP.log_omega),
        label=r"Ho-Bird-Garnett (2020) $\omega$",
        color="lightblue",
    )

    ax.legend()
    ax.set_xlabel(r"Rest-frame Wavelength $(\AA)$")
    ax.set_ylabel(r"Normalized Flux")
    ax.set_xlim([min_lambda, max_lambda])

    ax12 = ax.twiny()
    ax12.set_xticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ]
    )
    ax12.set_xticklabels(
        [r"Ly $\infty$", r"Ly $\beta$", r"Ly $\alpha$", r"NV", r"OI", r"CII", r"SIV"]
    )

    plt.tight_layout()
    save_figure("omega_changes")
    plt.clf()

    # plotting covariance matrix
    min_lambda = qsos_full_int.GP.min_lambda
    max_lambda = qsos_full_int.GP.max_lambda
    scale = np.shape(qsos_full_int.GP.C)[0] / (max_lambda - min_lambda)

    lyg_wavelength = 972

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(qsos_full_int.GP.C, origin="lower")
    ax.set_xticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyg_wavelength - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ],
    )
    ax.set_xticklabels(
        [
            r"Ly $\infty$",
            r"Ly $\gamma$",
            r"Ly $\beta$",
            r"Ly $\alpha$",
            r"NV",
            r"OI",
            r"CII",
            r"SIV",
        ],
        rotation=45,
    )
    ax.set_yticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyg_wavelength - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ]
    )
    ax.set_yticklabels(
        [
            r"Ly $\infty$",
            r"Ly $\gamma$",
            r"Ly $\beta$",
            r"Ly $\alpha$",
            r"NV",
            r"OI",
            r"CII",
            r"SIV",
        ]
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_figure("covariance_matrix")
    plt.clf()


def do_cddf_occams(
    subdir: str = "CDDF_analysis",
    subdir_occams_upper: str = "CDDF_analysis",
    subdir_occams_lower: str = "CDDF_analysis",
    label_upper: str = "GP Occams upper",
    label_lower: str = "GP Occams lower",
    outfile: str = "cddf_gp_occams",
    color_upper: str = "C1",
    color_lower: str = "C2",
    plot_gp: bool = True,
    plot_n12: bool = False,
):
    """
    Make CDDF plots with systematics from Occams razors.
    """
    if plot_n12:
        dla_data.noterdaeme_12_data()

    # (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]) in (6, N) shape
    if plot_gp:
        cddf_all = np.loadtxt(os.path.join(subdir, "cddf_all.txt"))
        (_, N) = cddf_all.shape

        cddf68 = np.full((N, 2), fill_value=np.nan)
        cddf95 = np.full((N, 2), fill_value=np.nan)
        (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]) = cddf_all

        plot_cddf(l_N, cddf, cddf68, cddf95)

    # Occams lower and upper
    cddf_all = np.loadtxt(os.path.join(subdir_occams_upper, "cddf_all.txt"))
    (_, N) = cddf_all.shape

    cddf68 = np.full((N, 2), fill_value=np.nan)
    cddf95 = np.full((N, 2), fill_value=np.nan)
    (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]) = cddf_all

    plot_cddf(l_N, cddf, cddf68, cddf95, label=label_upper, color=color_upper)

    # lower
    cddf_all = np.loadtxt(os.path.join(subdir_occams_lower, "cddf_all.txt"))
    (_, N) = cddf_all.shape

    cddf68 = np.full((N, 2), fill_value=np.nan)
    cddf95 = np.full((N, 2), fill_value=np.nan)
    (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]) = cddf_all

    plot_cddf(l_N, cddf, cddf68, cddf95, label=label_lower, color=color_lower)
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    plt.tight_layout()
    save_figure(os.path.join(subdir, outfile))
    plt.clf()


def plot_cddf(
    l_N: np.ndarray,
    cddf: np.ndarray,
    cddf68: np.ndarray,
    cddf95: np.ndarray,
    lnhi_nbins: int = 30,
    lnhi_min: float = 20.0,
    lnhi_max: float = 23.0,
    color="blue",
    label="GP",
):
    """
    Plot cddf from variables, using same methods as calc_cddf.plot_cddf
    """
    # Get the NHI bins
    l_nhi = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)

    xerrs = (10 ** l_N - 10 ** l_nhi[:-1], 10 ** l_nhi[1:] - 10 ** l_N)

    # two sigma
    plt.fill_between(10 ** l_N, cddf95[:, 0], cddf95[:, 1], color="grey", alpha=0.5)

    # if has values
    yerr = (cddf - cddf68[:, 0], cddf68[:, 1] - cddf)
    ii = np.where(cddf68[:, 0] > 0.0)
    if np.size(ii) > 0:
        plt.errorbar(
            10 ** l_N[ii],
            cddf[ii],
            yerr=(yerr[0][ii], yerr[1][ii]),
            xerr=(xerrs[0][ii], xerrs[1][ii]),
            fmt="o",
            label=label,
            color=color,
        )

    # upper limits
    i2 = np.where(cddf68[:, 0] == 0)
    if np.size(i2) > 0:
        plt.errorbar(
            10 ** l_N[i2],
            cddf[i2] + yerr[1][i2],
            yerr=yerr[1][i2] / 2.0,
            xerr=(xerrs[0][i2], xerrs[1][i2]),
            fmt="o",
            label=None,
            uplims=True,
            color=color,
            lw=2,
        )
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
    plt.ylabel(r"$f(N_\mathrm{HI})$")


def do_dndx_occams(
    subdir: str = "CDDF_analysis",
    subdir_occams_upper: str = "CDDF_analysis",
    subdir_occams_lower: str = "CDDF_analysis",
    label_upper: str = "GP Occams upper",
    label_lower: str = "GP Occams lower",
    color_upper: str = "C1",
    color_lower: str = "C2",
    outfile: str = "dndx_gp_occams",
    plot_gp: bool = True,
):
    """
    Make dNdX plots with systematics from Occams razors.
    """
    dla_data.dndx_not()
    dla_data.dndx_pro()

    # (z_cent, dNdX, dndx68[:,0],dndx68[:,1], dndx95[:,0],dndx95[:,1]) in (6, N) shape
    if plot_gp:
        dndx_all = np.loadtxt(os.path.join(subdir, "dndx_all.txt"))
        (_, N) = dndx_all.shape

        dndx68 = np.full((N, 2), fill_value=np.nan)
        dndx95 = np.full((N, 2), fill_value=np.nan)
        (
            z_cent,
            dNdX,
            dndx68[:, 0],
            dndx68[:, 1],
            dndx95[:, 0],
            dndx95[:, 1],
        ) = dndx_all

        plot_line_density(z_cent, dNdX, dndx68, dndx95, color="C0")

    # Occams Upper
    dndx_all = np.loadtxt(os.path.join(subdir_occams_upper, "dndx_all.txt"))
    (_, N) = dndx_all.shape

    dndx68 = np.full((N, 2), fill_value=np.nan)
    dndx95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, dNdX, dndx68[:, 0], dndx68[:, 1], dndx95[:, 0], dndx95[:, 1]) = dndx_all

    plot_line_density(
        z_cent, dNdX, dndx68, dndx95, color=color_upper, label=label_upper
    )

    # Occams Lower
    dndx_all = np.loadtxt(os.path.join(subdir_occams_lower, "dndx_all.txt"))
    (_, N) = dndx_all.shape

    dndx68 = np.full((N, 2), fill_value=np.nan)
    dndx95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, dNdX, dndx68[:, 0], dndx68[:, 1], dndx95[:, 0], dndx95[:, 1]) = dndx_all

    plot_line_density(
        z_cent, dNdX, dndx68, dndx95, color=color_lower, label=label_lower
    )
    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    plt.tight_layout()
    save_figure(os.path.join(subdir, outfile))
    plt.clf()


def plot_line_density(
    z_cent: np.ndarray,
    dNdX: np.ndarray,
    dndx68: np.ndarray,
    dndx95: np.ndarray,
    zmin: float = 2.0,
    zmax: float = 5.0,
    label: str = "GP",
    color="blue",
    bins_per_z: int = 6,
):
    """Plot the line density as a function of redshift"""
    # Get the redshifts
    nbins = np.max([int((zmax - zmin) * bins_per_z), 1])
    z_bins = np.linspace(zmin, zmax, nbins + 1)

    try:
        # hope this line works
        xerrs = (z_cent - z_bins[:-1], z_bins[1:] - z_cent)
    except ValueError as e:
        xerrs = (z_cent - z_bins[:-2], z_bins[1:-1] - z_cent)

    # 2 sigma contours.
    plt.fill_between(z_cent, dndx95[:, 0], dndx95[:, 1], color="grey", alpha=0.5)
    yerr = (dNdX - dndx68[:, 0], dndx68[:, 1] - dNdX)
    plt.errorbar(z_cent, dNdX, yerr=yerr, xerr=xerrs, fmt="o", color=color, label=label)
    plt.xlabel(r"z")
    plt.ylabel(r"dN/dX")
    plt.xlim(zmin, zmax)


def do_omega_dla_occams(
    subdir: str = "CDDF_analysis",
    subdir_occams_upper: str = "CDDF_analysis",
    subdir_occams_lower: str = "CDDF_analysis",
    label_upper: str = "GP Occams upper",
    label_lower: str = "GP Occams lower",
    color_upper: str = "C1",
    color_lower: str = "C2",
    outfile: str = "omega_occams",
    plot_gp: bool = True,
    plot_pw09: bool = False,
    xq100_omega: bool = True,
):
    """
    Make OmegaDLA plots with systematics from Occams razors.
    """
    dla_data.omegahi_not()
    if plot_pw09:
        dla_data.omegahi_pro()
    dla_data.crighton_omega()
    if xq100_omega:
        dla_data.xq100_omega()

    # (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) in (6, N) shape
    if plot_gp:
        omega_dla_all = np.loadtxt(os.path.join(subdir, "omega_dla_all.txt"))
        (_, N) = omega_dla_all.shape

        omega_dla_68 = np.full((N, 2), fill_value=np.nan)
        omega_dla_95 = np.full((N, 2), fill_value=np.nan)
        (
            z_cent,
            omega_dla,
            omega_dla_68[:, 0],
            omega_dla_68[:, 1],
            omega_dla_95[:, 0],
            omega_dla_95[:, 1],
        ) = omega_dla_all

        plot_omega_dla(z_cent, omega_dla, omega_dla_68, omega_dla_95, color="C0")

    # Occams upper
    omega_dla_all = np.loadtxt(os.path.join(subdir_occams_upper, "omega_dla_all.txt"))
    (_, N) = omega_dla_all.shape

    omega_dla_68 = np.full((N, 2), fill_value=np.nan)
    omega_dla_95 = np.full((N, 2), fill_value=np.nan)
    (
        z_cent,
        omega_dla,
        omega_dla_68[:, 0],
        omega_dla_68[:, 1],
        omega_dla_95[:, 0],
        omega_dla_95[:, 1],
    ) = omega_dla_all

    plot_omega_dla(
        z_cent,
        omega_dla,
        omega_dla_68,
        omega_dla_95,
        color=color_upper,
        label=label_upper,
    )

    # Occams lower
    omega_dla_all = np.loadtxt(os.path.join(subdir_occams_lower, "omega_dla_all.txt"))
    (_, N) = omega_dla_all.shape

    omega_dla_68 = np.full((N, 2), fill_value=np.nan)
    omega_dla_95 = np.full((N, 2), fill_value=np.nan)
    (
        z_cent,
        omega_dla,
        omega_dla_68[:, 0],
        omega_dla_68[:, 1],
        omega_dla_95[:, 0],
        omega_dla_95[:, 1],
    ) = omega_dla_all

    plot_omega_dla(
        z_cent,
        omega_dla,
        omega_dla_68,
        omega_dla_95,
        color=color_lower,
        label=label_lower,
    )

    plt.legend(loc=0)
    plt.xlim(2, 5)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    save_figure(os.path.join(subdir, outfile))
    plt.clf()


def do_omega_dla_XQ100(subdir: str = "CDDF_analysis", plot_pw09: bool = False):
    """
    Make OmegaDLA plots with systematics from Occams razors.
    """
    dla_data.omegahi_not()
    if plot_pw09:
        dla_data.omegahi_pro()
    dla_data.crighton_omega()
    dla_data.xq100_omega()

    # (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) in (6, N) shape
    omega_dla_all = np.loadtxt(os.path.join(subdir, "omega_dla_all.txt"))
    (_, N) = omega_dla_all.shape

    omega_dla_68 = np.full((N, 2), fill_value=np.nan)
    omega_dla_95 = np.full((N, 2), fill_value=np.nan)
    (
        z_cent,
        omega_dla,
        omega_dla_68[:, 0],
        omega_dla_68[:, 1],
        omega_dla_95[:, 0],
        omega_dla_95[:, 1],
    ) = omega_dla_all

    plot_omega_dla(z_cent, omega_dla, omega_dla_68, omega_dla_95, color="C0")

    plt.legend(loc=0)
    plt.xlim(2, 5)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    save_figure(os.path.join(subdir, "omega_xq100"))
    plt.clf()


def plot_omega_dla(
    z_cent: np.ndarray,
    omega_dla: np.ndarray,
    omega_dla_68: np.ndarray,
    omega_dla_95: np.ndarray,
    zmin: float = 2.0,
    zmax: float = 5.0,
    label: str = "GP",
    color: str = "blue",
    twosigma: bool = True,
    bins_per_z: int = 6,
):
    """Plot omega_DLA as a function of redshift, with full Bayesian errors"""
    nbins = np.max([int((zmax - zmin) * bins_per_z), 1])
    z_bins = np.linspace(zmin, zmax, nbins + 1)

    try:
        # hope this line works
        xerrs = (z_cent - z_bins[:-1], z_bins[1:] - z_cent)
    except ValueError as e:
        xerrs = (z_cent - z_bins[:-2], z_bins[1:-1] - z_cent)

    # import pdb
    # pdb.set_trace()

    if twosigma:
        plt.fill_between(
            z_cent,
            1000 * omega_dla_95[:, 0],
            1000 * omega_dla_95[:, 1],
            color="grey",
            alpha=0.5,
        )
    yerr = (
        omega_dla - 1000 * omega_dla_68[:, 0],
        1000 * omega_dla_68[:, 1] - omega_dla,
    )
    plt.errorbar(
        z_cent, omega_dla, yerr=yerr, xerr=xerrs, fmt="s", label=label, color=color
    )
    plt.xlabel(r"z")
    plt.ylabel(r"$10^3 \times \Omega_\mathrm{DLA}$")
    plt.xlim(zmin, zmax)


def do_Parks_CDDF(
    qsos: QSOLoaderDR16Q,
    dla_cnn: str = "data/dr16q/distfiles/DR16Q_v4.fits", # "data/dr16q/distfiles/DLA_CAT_SDSS_DR16.fits"
    subdir: str = "CDDF_analysis/parks_cddf_dr16q/", # CDDF_analysis/solene_cddf_dr16q
    p_thresh: float = 0.98,
    snr_thresh: float = -2.0,
    solene_dlas: bool = False, # Solene only
    voigt_fitter: bool = False, # Solene only
    our_sightlines: bool = True, # Solene only
    lyb: bool = False,
    min_1040: bool = False,
    search_range_from_ours: bool = False,
    plot_gp: bool = True,
    subdir_gp: str = "CDDF_analysis/dr16q_full_int_lyb_occam_zqso7_1_30_delta_z_0_1/",
):
    """
    Plot the column density function of Parks (2018) or Solene (2021)

    Parameters:
    ----
    dla_cnn (str) : path to Parks' CNN model in the DR16Q
    """

    pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)

    dla_data.noterdaeme_12_data()
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_cnn,
        zmax=5,
        color="C4",
        p_thresh=p_thresh,
        snr_thresh=snr_thresh,
        apply_p_dlas=False,
        lyb=lyb,
        min_1040=min_1040,
        search_range_from_ours=search_range_from_ours,
        label="CNN",
        solene_dlas=solene_dlas,
        voigt_fitter=voigt_fitter,
        our_sightlines=our_sightlines,
    )

    # set the specialized figure names, and double check the args
    if solene_dlas:
        assert "dict_solene" in dir(qsos)
        figname = "solene"
    else:
        assert ~voigt_fitter
        figname = "parks"

    np.savetxt(os.path.join(subdir, "cddf_{}_all.txt".format(figname)), (l_N, cddf))

    # plot alongside with GP model
    if plot_gp:
        cddf_all = np.loadtxt(os.path.join(subdir_gp, "cddf_all.txt"))
        (_, N) = cddf_all.shape

        cddf68 = np.full((N, 2), fill_value=np.nan)
        cddf95 = np.full((N, 2), fill_value=np.nan)
        (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]) = cddf_all

        plot_cddf(l_N, cddf, cddf68, cddf95)

    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    plt.tight_layout()
    save_figure(os.path.join(subdir, "cddf_{}".format(figname)))
    plt.clf()

    # Evolution with redshift
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_cnn,
        zmin=4,
        zmax=5,
        label="4-5",
        color="brown",
        p_thresh=p_thresh,
        snr_thresh=snr_thresh,
        apply_p_dlas=False,
        lyb=lyb,
        min_1040=min_1040,
        search_range_from_ours=search_range_from_ours,
        solene_dlas=solene_dlas,
        voigt_fitter=voigt_fitter,
        our_sightlines=our_sightlines,
    )
    np.savetxt(os.path.join(subdir, "cddf_{}_z45.txt".format(figname)), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_cnn,
        zmin=3,
        zmax=4,
        label="3-4",
        color="black",
        p_thresh=p_thresh,
        snr_thresh=snr_thresh,
        apply_p_dlas=False,
        lyb=lyb,
        min_1040=min_1040,
        search_range_from_ours=search_range_from_ours,
        solene_dlas=solene_dlas,
        voigt_fitter=voigt_fitter,
        our_sightlines=our_sightlines,
    )
    np.savetxt(os.path.join(subdir, "cddf_{}_z34.txt".format(figname)), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_cnn,
        zmin=2.5,
        zmax=3,
        label="2.5-3",
        color="green",
        p_thresh=p_thresh,
        snr_thresh=snr_thresh,
        apply_p_dlas=False,
        lyb=lyb,
        min_1040=min_1040,
        search_range_from_ours=search_range_from_ours,
        solene_dlas=solene_dlas,
        voigt_fitter=voigt_fitter,
        our_sightlines=our_sightlines,
    )
    np.savetxt(os.path.join(subdir, "cddf_{}_z253.txt".format(figname)), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_cnn,
        zmin=2,
        zmax=2.5,
        label="2-2.5",
        color="blue",
        p_thresh=p_thresh,
        snr_thresh=snr_thresh,
        apply_p_dlas=False,
        lyb=lyb,
        min_1040=min_1040,
        search_range_from_ours=search_range_from_ours,
        solene_dlas=solene_dlas,
        voigt_fitter=voigt_fitter,
        our_sightlines=our_sightlines,
    )
    np.savetxt(os.path.join(subdir, "cddf_{}_z225.txt".format(figname)), (l_N, cddf))

    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    plt.tight_layout()
    save_figure(os.path.join(subdir, "cddf_zz_{}".format(figname)))
    plt.clf()


def do_Parks_dNdX(
    qsos: QSOLoaderDR16Q,
    dla_cnn: str = "data/dr16q/distfiles/DR16Q_v4.fits", # "data/dr16q/distfiles/DLA_CAT_SDSS_DR16.fits"
    subdir: str = "CDDF_analysis/parks_cddf_dr16q/", # CDDF_analysis/solene_cddf_dr16q
    p_thresh: float = 0.98,
    snr_thresh: float = -2.0,
    solene_dlas: bool = False, # Solene only
    voigt_fitter: bool = False, # Solene only
    our_sightlines: bool = True, # Solene only
    lyb: bool = False,
    min_1040: bool = False,
    search_range_from_ours: bool = False,
    plot_gp: bool = True,
    subdir_gp: str = "CDDF_analysis/dr16q_full_int_lyb_occam_zqso7_1_30_delta_z_0_1/",
    lnhi_min: float = 20.3,
):
    """
    Plot dNdX for Parks' CNN model in the DR16Q

    Parameters:
    ----
    dla_cnn (str) : path to Parks' CNN model in the DR16Q
    """

    pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)

    dla_data.dndx_not()
    dla_data.dndx_pro()
    z_cent, dNdX = qsos.plot_line_density_park(
        dla_cnn,
        zmax=5,
        p_thresh=p_thresh,
        snr_thresh=snr_thresh,
        apply_p_dlas=False,
        lyb=lyb,
        min_1040=min_1040,
        search_range_from_ours=search_range_from_ours,
        color="C4",
        label="CNN",
        solene_dlas=solene_dlas,
        voigt_fitter=voigt_fitter,
        our_sightlines=our_sightlines,
        lnhi_min=lnhi_min,
    )
    np.savetxt(os.path.join(subdir, "dndx_all.txt"), (z_cent, dNdX))

    # set the specialized figure names, and double check the args
    if solene_dlas:
        assert "dict_solene" in dir(qsos)
        figname = "solene"
    else:
        assert ~voigt_fitter
        figname = "parks"

    # (z_cent, dNdX, dndx68[:,0],dndx68[:,1], dndx95[:,0],dndx95[:,1]) in (6, N) shape
    if plot_gp:
        dndx_all = np.loadtxt(os.path.join(subdir_gp, "dndx_all.txt"))
        (_, N) = dndx_all.shape

        dndx68 = np.full((N, 2), fill_value=np.nan)
        dndx95 = np.full((N, 2), fill_value=np.nan)
        (
            z_cent,
            dNdX,
            dndx68[:, 0],
            dndx68[:, 1],
            dndx95[:, 0],
            dndx95[:, 1],
        ) = dndx_all

        plot_line_density(z_cent, dNdX, dndx68, dndx95, color="C0")

    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    plt.tight_layout()
    save_figure(os.path.join(subdir, "dndx_{}".format(figname)))
    plt.clf()


def do_Parks_OmegaDLA(
    qsos: QSOLoaderDR16Q,
    dla_cnn: str = "data/dr16q/distfiles/DR16Q_v4.fits", # "data/dr16q/distfiles/DLA_CAT_SDSS_DR16.fits"
    subdir: str = "CDDF_analysis/parks_cddf_dr16q/", # CDDF_analysis/solene_cddf_dr16q
    zmin: float = 2.0,
    zmax: float = 4.0,
    p_thresh: float = 0.98,
    snr_thresh: float = -2.0,
    solene_dlas: bool = False, # Solene only
    voigt_fitter: bool = False, # Solene only
    our_sightlines: bool = True, # Solene only
    lyb: bool = False,
    min_1040: bool = False,
    search_range_from_ours: bool = False,
    plot_gp: bool = True,
    subdir_gp: str = "CDDF_analysis/dr16q_full_int_lyb_occam_zqso7_1_30_delta_z_0_1/",
):
    """
    Plot OmegaDLA for Parks's CNN model in DR16Q
    """

    pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)

    # Omega_DLA
    dla_data.omegahi_not()
    dla_data.omegahi_pro()
    dla_data.crighton_omega()

    (z_cent, omega_dla) = qsos.plot_omega_dla_parks(
        dla_cnn,
        zmin=zmin,
        zmax=zmax,
        p_thresh=p_thresh,
        snr_thresh=snr_thresh,
        lyb=lyb,
        min_1040=min_1040,
        search_range_from_ours=search_range_from_ours,
        color="C4",
        label="CNN",
        solene_dlas=solene_dlas,
        voigt_fitter=voigt_fitter,
        our_sightlines=our_sightlines,
    )

    np.savetxt(os.path.join(subdir, "omega_dla_all.txt"), (z_cent, omega_dla))

    # set the specialized figure names, and double check the args
    if solene_dlas:
        assert "dict_solene" in dir(qsos)
        figname = "solene"
    else:
        assert ~voigt_fitter
        figname = "parks"

    # (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) in (6, N) shape
    if plot_gp:
        omega_dla_all = np.loadtxt(os.path.join(subdir_gp, "omega_dla_all.txt"))
        (_, N) = omega_dla_all.shape

        omega_dla_68 = np.full((N, 2), fill_value=np.nan)
        omega_dla_95 = np.full((N, 2), fill_value=np.nan)
        (
            z_cent,
            omega_dla,
            omega_dla_68[:, 0],
            omega_dla_68[:, 1],
            omega_dla_95[:, 0],
            omega_dla_95[:, 1],
        ) = omega_dla_all

        plot_omega_dla(z_cent, omega_dla, omega_dla_68, omega_dla_95, color="C0")

    plt.legend(loc=0)
    plt.xlim(2, 5)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    save_figure(os.path.join(subdir, "omega_{}".format(figname)))
    plt.clf()


def do_Parks_snr_check(
    qsos: QSOLoaderDR16Q,
    dla_cnn: str = "data/dr16q/distfiles/DR16Q_v4.fits",
    subdir: str = "CDDF_analysis/parks_cddf_dr16q/",
    p_thresh: float = 0.98,
    lyb: bool = False,
    search_range_from_ours: bool = False,
):
    """
    Check effect of removing spectra with low SNRs.
    """
    snrs_list = (-2, 2, 4, 8)

    # CDDF
    dla_data.noterdaeme_12_data()
    for i, snr_thresh in enumerate(snrs_list):
        (l_N, cddf) = qsos.plot_cddf_parks(
            dla_cnn,
            zmax=5,
            p_thresh=p_thresh,
            color=cmap((i + 1) / len(snrs_list)),
            snr_thresh=snr_thresh,
            label="Parks SNR > {:d}".format(snr_thresh),
            apply_p_dlas=False,
            lyb=lyb,
            search_range_from_ours=search_range_from_ours,
        )

    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_parks_snr"))
    plt.clf()

    # dN/dX
    dla_data.dndx_not()
    dla_data.dndx_pro()
    for i, snr_thresh in enumerate(snrs_list):
        z_cent, dNdX = qsos.plot_line_density_park(
            dla_cnn,
            zmax=5,
            p_thresh=p_thresh,
            color=cmap((i + 1) / len(snrs_list)),
            snr_thresh=snr_thresh,
            label="Parks SNR > {:d}".format(snr_thresh),
            apply_p_dlas=False,
            lyb=lyb,
            search_range_from_ours=search_range_from_ours,
        )

    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    save_figure(os.path.join(subdir, "dndx_parks_snr"))
    plt.clf()


def do_confusion_parks(
    qsos: QSOLoaderDR16Q,
    dla_cnn: str = "data/dr16q/distfiles/DR16Q_v4.fits",
    snr: float = -1.0,
    dla_confidence: float = 0.98,
    p_thresh: float = 0.98,
    lyb: bool = True,
):
    """
    plot the multi-DLA confusion matrix between our MAP predictions and Parks' predictions
    """
    if "dla_catalog_parks" not in dir(qsos):
        qsos.load_dla_parks(
            dla_cnn, p_thresh=dla_confidence, multi_dla=False, num_dla=1
        )

    confusion_matrix, _ = qsos.make_multi_confusion(
        qsos.dla_catalog_parks, dla_confidence, p_thresh, snr=snr, lyb=lyb
    )

    size, _ = confusion_matrix.shape

    print("Confusion Matrix Garnett's Multi-DLA versus Parks : ")
    print("----")
    for i in range(size):
        print("{} DLA".format(i), end="\t")
        for j in range(size):
            print("{}".format(confusion_matrix[i, j]), end=" ")
        print("")

    print("Mutli-DLA disagreements : ")
    print("----")
    for i in range(size):
        num = (
            confusion_matrix[(i + 1) :, 0 : (i + 1)].sum()
            + confusion_matrix[0 : (i + 1), (i + 1) :].sum()
        )
        print(
            "Error between >= {} DLAs and < {} DLAs: {:.2g}".format(
                i + 1, i + 1, num / confusion_matrix.sum()
            )
        )


def do_MAP_parks_comparison(
    qsos: QSOLoaderDR16Q,
    dla_cnn: str = "data/dr16q/distfiles/DR16Q_v4.fits",
    dla_confidence: float = 0.98,
    num_dlas: int = 1,
    num_bins: int = 100,
):
    """
    Plot the comparisons between MAP values and Parks' predictions
    """
    if "dla_catalog_parks" not in dir(qsos):
        qsos.load_dla_parks(
            dla_cnn, p_thresh=dla_confidence, multi_dla=False, num_dla=1
        )

    Delta_z_dlas, Delta_log_nhis, z_dlas_parks = qsos.make_MAP_parks_comparison(
        qsos.dla_catalog_parks, num_dlas=num_dlas, dla_confidence=dla_confidence
    )

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    # plot in the same scale as in the Garnett (2017) paper
    # for z_dlas, xlim(-0.01, 0.01); for log_nhis, xlim(-1, 1)
    # TODO: make KDE plot
    axs[0].hist(
        Delta_z_dlas.ravel(), bins=np.linspace(-0.01, 0.01, num_bins), density=True
    )
    axs[0].set_xlim(-0.01, 0.01)
    axs[0].set_xlabel(
        r"difference between $z_{DLA}$ (new code/Parks);" + " DLA({})".format(num_dlas)
    )
    axs[0].set_ylabel(r"$p$(difference)")

    axs[1].hist(Delta_log_nhis.ravel(), bins=np.linspace(-1, 1, num_bins), density=True)
    axs[1].set_xlim(-1, 1)
    axs[1].set_xlabel(
        r"difference between $\log{N_{HI}}$ (new code/Parks);"
        + " DLA({})".format(num_dlas)
    )
    axs[1].set_ylabel(r"$p$(difference)")

    plt.tight_layout()
    save_figure(
        "MAP_comparison_Parks_dlas{}_confidence{}".format(
            num_dlas, str(dla_confidence).replace(".", "_")
        )
    )
    plt.clf()
    plt.close()

    high_z = np.array([2.5, 3.0, 3.5, 5.0])
    low_z = np.array([2.0, 2.5, 3.0, 3.5])
    log_nhi_evo = []
    sterr_log_nhi_evo = []
    z_dla_evo = []
    sterr_z_dla_evo = []
    for high_z_dla, low_z_dla in zip(high_z, low_z):
        inds = (z_dlas_parks > low_z_dla) & (z_dlas_parks < high_z_dla)

        z_dla_evo.append(np.nanmean(Delta_z_dlas[inds]))
        log_nhi_evo.append(np.nanmean(Delta_log_nhis[inds]))

        # stderr = sqrt(s / n)
        sterr_z_dla_evo.append(
            np.sqrt(
                np.nansum((Delta_z_dlas[inds] - np.nanmean(Delta_z_dlas[inds])) ** 2)
                / (Delta_z_dlas[inds].shape[0] - 1)
            )
            / np.sqrt(Delta_z_dlas[inds].shape[0])
        )

        sterr_log_nhi_evo.append(
            np.sqrt(
                np.nansum(
                    (Delta_log_nhis[inds] - np.nanmean(Delta_log_nhis[inds])) ** 2
                )
                / (Delta_log_nhis[inds].shape[0] - 1)
            )
            / np.sqrt(Delta_log_nhis[inds].shape[0])
        )

    z_cent = np.array([(z_x + z_m) / 2.0 for (z_m, z_x) in zip(high_z, low_z)])
    xerrs = (z_cent - low_z, high_z - z_cent)

    plt.errorbar(
        z_cent,
        z_dla_evo,
        yerr=sterr_z_dla_evo,
        xerr=xerrs,
        label=r"$z_{{DLA}} - z_{{DLA}}^{{Parks}} \mid \mathrm{{Garnett}}({k}) \cap \mathrm{{Parks}}({k})$".format(
            k=num_dlas
        ),
    )
    plt.xlabel(r"$z_{DLA}^{Parks}$")
    plt.ylabel(r"$\Delta z_{DLA}$")
    plt.legend(loc=0)
    save_figure("MAP_z_dla_evolution_parks")
    plt.clf()

    plt.errorbar(
        z_cent,
        log_nhi_evo,
        yerr=sterr_log_nhi_evo,
        xerr=xerrs,
        label=r"$\log{{N_{{HI}}}}_{{DLA}} - \log{{N_{{HI}}}}_{{DLA}}^{{Parks}} \mid \mathrm{{Garnett}}({k}) \cap \mathrm{{Parks}}({k})$".format(
            k=num_dlas
        ),
    )
    plt.xlabel(r"$z_{DLA}^{Parks}$")
    plt.ylabel(r"$\Delta \log{N_{HI}}$")
    plt.legend(loc=0)
    save_figure("MAP_log_nhi_evolution_parks")
    plt.clf()


def do_MAP_hist2d_parks(
    qsos: QSOLoaderDR16Q,
    dla_cnn: str = "data/dr16q/distfiles/DR16Q_v4.fits",
    dla_confidence: float = 0.98,
    num_dlas: int = 1,
):
    """
    Do the hist2d in between z_true vs z_map
    """
    if "dla_catalog_parks" not in dir(qsos):
        qsos.load_dla_parks(
            dla_cnn, p_thresh=dla_confidence, multi_dla=False, num_dla=1
        )

    (
        _,
        _,
        _,
        gp_z_dlas,
        gp_log_nhis,
        cnn_z_dlas,
        cnn_log_nhis,
    ) = qsos.make_MAP_parks_comparison(
        qsos.dla_catalog_parks,
        num_dlas=num_dlas,
        dla_confidence=dla_confidence,
        return_map_values=True,
    )

    # expand the arrays
    gp_z_dlas = gp_z_dlas.ravel()
    gp_log_nhis = gp_log_nhis.ravel()
    cnn_z_dlas = cnn_z_dlas.ravel()
    cnn_log_nhis = cnn_log_nhis.ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    (h1, x1edges, y1edges, im1) = ax1.hist2d(
        gp_z_dlas, cnn_z_dlas, bins=int(np.sqrt(gp_z_dlas.shape[0])), cmap="viridis"
    )
    # a perfect prediction straight line
    z_dlas_plot = np.linspace(2.0, 5.0, 100)
    ax1.plot(z_dlas_plot, z_dlas_plot)
    ax1.set_xlabel(r"$z_{{DLA,MAP}}$")
    ax1.set_ylabel(r"$z_{{DLA,CNN}}$")
    fig.colorbar(im1, ax=ax1)

    (h2, x2edges, y2edges, im2) = ax2.hist2d(
        gp_log_nhis,
        cnn_log_nhis,
        bins=int(np.sqrt(gp_log_nhis.shape[0])),
        cmap="viridis",
    )

    # a perfect prediction straight line
    log_nhi_plot = np.linspace(20, 22.5, 100)
    ax2.plot(log_nhi_plot, log_nhi_plot)

    # # 3rd polynomial fit
    # poly_fit =  np.poly1d( np.polyfit(map_log_nhis, true_log_nhis, 4 ) )
    # ax2.plot(log_nhi_plot, poly_fit(log_nhi_plot), color="white", ls='--')

    ax2.set_xlabel(r"$\log N_{{HI,MAP}}$")
    ax2.set_ylabel(r"$\log N_{{HI,CNN}}$")
    # ax2.set_xlim(20, 22.5)
    # ax2.set_ylim(20, 22.5)
    fig.colorbar(im2, ax=ax2)

    print(
        "Pearson Correlation for (map_z_dlas,   cnn_z_dlas) : ",
        pearsonr(gp_z_dlas, cnn_z_dlas),
    )
    print(
        "Pearson Correlation for (map_log_nhis, cnn_log_nhis) : ",
        pearsonr(gp_log_nhis, cnn_log_nhis),
    )

    # examine the pearson correlation per log nhi bins
    log_nhi_bins = [20, 20.5, 21, 23]

    for (min_log_nhi, max_log_nhi) in zip(log_nhi_bins[:-1], log_nhi_bins[1:]):
        ind = (gp_log_nhis > min_log_nhi) & (gp_log_nhis < max_log_nhi)
        ind = ind & (cnn_log_nhis > min_log_nhi) & (cnn_log_nhis < max_log_nhi)

        print(
            "Map logNHI Bin [{}, {}] Pearson Correlation for (map_log_nhis, true_log_nhi) : ".format(
                min_log_nhi, max_log_nhi
            ),
            pearsonr(gp_log_nhis[ind], cnn_log_nhis[ind]),
        )

    save_figure("MAP_hist2d_GP_CNN")


def do_plot_this_mu_with_labels(
    nspec: int,
    nspec_original: int,
    qsos_dr16q: QSOLoaderDR16Q,
    qsos_dr16q_original: QSOLoaderDR16Q,
    if_original: bool = True,
):
    """
    Plot this mu comparing between marginalising the meanflux uncertainty or without marginalising,
    with shaded area on [Lyb, Lya]
    """
    qsos_dr16q.plot_this_mu(
        nspec, new_fig=True, Parks=True, label="marginalizing meanflux: "
    )
    if if_original:
        qsos_dr16q_original.plot_this_mu(
            nspec_original,
            new_fig=False,
            label="without marginalizing meanflux: ",
            color="C9",
        )

    rest_wavelengths = qsos_dr16q.GP.rest_wavelengths
    min_rest_wavelength = (
        (1 + qsos_dr16q.min_z_dlas[nspec])
        * lya_wavelength
        / (1 + qsos_dr16q.z_qsos[nspec])
    )
    max_rest_wavelength = (
        (1 + qsos_dr16q.max_z_dlas[nspec])
        * lya_wavelength
        / (1 + qsos_dr16q.z_qsos[nspec])
    )
    ind = (rest_wavelengths > min_rest_wavelength) * (
        rest_wavelengths < max_rest_wavelength
    )

    plt.fill_between(rest_wavelengths[ind], -1, 5, alpha=0.25, color="grey")

    plt.ylim(-1, 5)
    plt.title(
        "thing ID = {}; z = {:.4g}, zPCA = {:.4g}, zPIPE = {:.4g}, zVI = {:.4g}; source_z = {}; CLASS_PERSON = {} (30: BALQSO, 50: Blazar)".format(
            qsos_dr16q.thing_ids[nspec],
            qsos_dr16q.z_qsos[nspec],
            qsos_dr16q.hdu[1].data["Z_PCA"][qsos_dr16q.test_real_index][nspec],
            qsos_dr16q.hdu[1].data["Z_PIPE"][qsos_dr16q.test_real_index][nspec],
            qsos_dr16q.hdu[1].data["Z_VI"][qsos_dr16q.test_real_index][nspec],
            qsos_dr16q.hdu[1].data["SOURCE_Z"][qsos_dr16q.test_real_index][nspec],
            qsos_dr16q.hdu[1].data["CLASS_PERSON"][qsos_dr16q.test_real_index][nspec],
        )
    )
    plt.tight_layout()
    save_figure("this_mu_{}".format(qsos_dr16q.thing_ids[nspec]))
    plt.show()

def plot_this_mu_overlapping_dlas(qsos: QSOLoaderDR16Q, Parks: bool = True, Solene: bool = True):
    """
    Plot the comparison of Parks, Solene, ours on spectra with solid DLA detections
    but overlapping/nearby DLAs.
    """
    selected_thing_ids = [
        147946665,
        177872172,
        194921144,
        209755715,
        225460412,
        229232104,
        236392107,
        248088771,
        268065713,
        269549832,
        296660604,
        307451922,
        332071702,
        349105103,
        371695814,
        37208460,
        416180602,
        473203069,
        487498804,
        488562798,
        501293410,
        540640951,
        54806980,
        60605850,
    ]

    all_specs = np.where(np.isin(qsos.thing_ids, selected_thing_ids))[0]

    for nspec in all_specs:
        qsos.plot_this_mu(nspec, Parks=Parks, Solene=Solene, voigt_fitter=False, dla_solene="data/dr16q/distfiles/DLA_CAT_SDSS_DR16.fits")
        plt.ylim(-1, 5)
        save_figure("plot_specs/CNN_overlaps/this_mu_{}".format(qsos.thing_ids[nspec]))
        plt.show()
