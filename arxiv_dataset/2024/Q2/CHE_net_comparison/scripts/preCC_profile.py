import numpy as np
import matplotlib.pyplot as plt
from MESAreader import getSrcCol, Rsun_cm, Msun, clight, G_cgs
import matplotlib.gridspec as gridspec
from glob import glob
import math

def calculate_lso_radius_and_spin_parameter(mass_bh, spin_parameter):
    """
    Calculate the radius and specific angular momentum for the marginally stable circular orbit
    around a Kerr black hole.

    Args:
        mass_bh (float): Mass of the black hole (in Msun).
        spin_parameter (float): Angular momentum a per unit mass (0 <= a <= 1).

    Returns:
        tuple: A tuple containing (r_lso_km, j_lso_cm2s).
            r_lso_cm (float): Radius of the marginally stable circular orbit (in km).
            j_lso_cm2s (float): Specific angular momentum (in cm^2/s).
    """
    # if not (0 <= spin_parameter <= 1):
    #     raise ValueError("Kerr parameter 'a' must be in the range [0, 1].")

    spin_parameter = abs(np.minimum(spin_parameter, 1.0))
    a = spin_parameter

    XX = (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    Z1 = 1.0 + XX * (1.0 - a ** 2.0) ** (1.0 / 3.0)
    Z2 = (3.0 * a ** 2.0 + Z1 ** 2.0) ** 0.5
    YY = ((3.0 - Z1) * (3.0 + Z1 + 2 * Z2)) ** 0.5

    r_lso1 = mass_bh * (3.0 + Z2 - YY)  # Prograde Orbit
    r_lso2 = mass_bh * (3.0 + Z2 + YY)  # Retrograde Orbit

    r_lso = r_lso1  # Only consider prograde (direct) orbits

    j_lso = r_lso * r_lso * mass_bh ** 0.5 / (r_lso ** (3.0 / 2.0) + a * mass_bh ** (3.0 / 2.0))
    j_lso = j_lso * Msun * G_cgs / clight  # Convert to cm^2/s
    r_lso = r_lso1 * Msun * G_cgs / (clight * clight)  # Convert to cm

    return r_lso, j_lso

def get_local_free_fall_timescale(pfile, bound_only=False):
    """
    returns the local free fall timescale as estimated by MESA:
    tau_ff = 2*pi*sqrt(r^3/(G*m))
    """
    src, col = getSrcCol(pfile)
    if bound_only:
        v = src[:, col.index("velocity")]
        vesc = src[:, col.index("vesc")]
        ind = v <= vesc
        m = src[ind, col.index("mass")] * Msun
        r = src[ind, col.index("radius")] * Rsun_cm
    else:
        m = src[:, col.index("mass")] * Msun
        r = src[:, col.index("radius")] * Rsun_cm
    tau_ff = 2 * math.pi * np.sqrt((r * r * r) / (G_cgs * m))
    return tau_ff  # in sec

def get_local_B_groth_timescale(pfile):
    """The Spruit-Tayler instability has an associated timescale
    \omega/\omega_A^2 and a lengthscale L_ST <= r\Omega/N with N
    Brunt-Vaisala -- from Spruit 1999 and Ji, Fuller et al. 2022
    """
    t_alfven = get_local_alfven_timescale(pfile) # sec
    src, col = getSrcCol(pfile)
    omega = src[:, col.index("omega")]
    N = np.sqrt(np.absolute(src[:, col.index("brunt_N2")]))
    index = (N == 0)
    N[index] = 1e-99
    t_B_growth = t_alfven*omega/N
    return t_B_growth # same unit as t_alfven

def get_local_alfven_timescale(pfile):
    src, col = getSrcCol(pfile)
    logbr = src[:, col.index("dynamo_log_B_r")]
    br = (10.0**logbr)
    zeroB = logbr <= 1e-50
    # logbphi = src[:, col.index("dynamo_log_B_phi")]
    r = src[:, col.index("radius")]*Rsun_cm
    rho = 10.0**(src[:, col.index("logRho")])
    va_r = br/np.sqrt(4*math.pi*rho)
    tau_alfven = r/va_r
    tau_alfven[zeroB] = -1 # set to negative value, no B_r means no Alfven waves
    return tau_alfven # in sec


def summary_pfile_plot_radius(pfile, fig_name=None):
    src, col = getSrcCol(pfile)
    logr = np.log10((10.**(src[:, col.index("logR")]))*Rsun_cm)
    logj = np.log10(src[:, col.index("omega")]*src[:, col.index("i_rot")])
    logbr = src[:, col.index("dynamo_log_B_r")]
    logbphi = src[:, col.index("dynamo_log_B_phi")]
    m = src[:, col.index("mass")]
    i3 = np.argmin(np.absolute(m-3.))

    v = src[:, col.index("velocity")]*1e-5 # km/s
    fig = plt.figure(figsize=(8,16))
    gs = gridspec.GridSpec(100, 100)
    ax = fig.add_subplot(gs[:, :])
    bx = ax.twinx()
    bx.set_ylim(-1000, 100)
    ax.axvline(logr[i3], 0,1, zorder=0, ls='--', c='k', lw=2)
    # ax.plot(logr, logj, label=r"$j = \omega I$")
    # ax.plot(logr, log_jisco, label=r"$J_\mathrm{ISCO}(a=0)$")
    ax.plot(logr, logbr, label=r"$B_r$")
    ax.plot(logr, logbphi, label=r"$B_\varphi$")
    bx.plot(logr, v, c='g', label=r"$v \ \mathrm{[km\ s^{-1}]}$")
    ax.legend(ncol=2, columnspacing=0.5, handlelength=0.5)
    ax.set_ylim(ymin=-5)
    # ax.set_ylim(10, 20)
    ax.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
    ax.set_ylabel(r"$\log_{10}(B/[\mathrm{G}])$")
    bx.set_ylabel(r"$v \ \mathrm{[km\ s^{-1}]}$")
    if fig_name:
        plt.savefig(fig_name)
    else:
        plt.show()


def plot_spin_tff_talfven_vs_radius(pfile, fig_name=None, title=None):
    src, col = getSrcCol(pfile)
    logr = np.log10((10.**(src[:, col.index("logR")]))*Rsun_cm)
    # logJinside = src[:, col.index("log_J_inside")]
    # Jinside = 10.** logJinside

    r = 10.0**logr
    omega = src[:, col.index("omega")]
    j_specific = r*r*omega
    # Jinside = np.cumsum(10.0**logj)
    mass = src[:, col.index("mass")]
    a = clight*j_specific/(G_cgs*mass*Msun)
    r_ISCO, j_ISCO = calculate_lso_radius_and_spin_parameter(mass, a)
    r_ISCO_nonrot, j_ISCO_nonrot = calculate_lso_radius_and_spin_parameter(mass, np.zeros(len(mass)))
    r_ISCO_maxrot, j_ISCO_maxrot = calculate_lso_radius_and_spin_parameter(mass, np.ones(len(mass)))
    # calculate dynamical timescale
    t_ff = get_local_free_fall_timescale(pfile)
    # calculate radial component of Alfven velocity
    t_alfv = get_local_alfven_timescale(pfile)

    fig = plt.figure(figsize=(8,16))
    gs = gridspec.GridSpec(150, 100)
    ax = fig.add_subplot(gs[:30, :])
    bx = fig.add_subplot(gs[30:60, :])
    cx = fig.add_subplot(gs[60:90, :])
    dx = fig.add_subplot(gs[90:150, :])

    ax.set_ylim(0,1)
    ax.set_ylabel(r"$a = cJ/GM^2$")
    bx.set_ylabel(r"$\tau_{\rm ff}$ [s]")
    bx.set_yscale('log')
    cx.set_ylabel(r"$\tau_{\rm Alfv}$ [s]")
    cx.set_ylim(1e-1, 1e13)
    cx.set_yscale('log')
    dx.set_yscale('log')
    dx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
    dx.plot(logr, j_ISCO_nonrot, c='k', ls=':', label=r"$j_{\rm ISCO}(a=0)$")
    dx.plot(logr, j_ISCO, c='k', ls='-', label=r"$j_{\rm ISCO}(a(t))$")
    dx.plot(logr, j_ISCO_maxrot, c='k', ls='--', label=r"$j_\mathrm{ISCO}(a=1)$")
    dx.plot(logr, j_specific, label=r"$j=r^2\omega$")
    dx.legend(fontsize=20, loc="lower right")
    dx.set_ylabel(r"$j \ [\mathrm{cm^2\ s^{-1}}]$")
    ax.plot(logr, a)
    bx.plot(logr, t_ff)
    cx.plot(logr, t_alfv)
    fig.align_ylabels()
    if title != None:
        ax.set_title(title, fontsize=30)
    if fig_name:
        plt.savefig(fig_name)
        print(fig_name)
    else:
        plt.show()


def plot_spin_tff_talfven_vs_mass(pfile, fig_name=None, title=None):
    src, col = getSrcCol(pfile)
    logr = np.log10((10.**(src[:, col.index("logR")]))*Rsun_cm)
    # logJinside = src[:, col.index("log_J_inside")]
    # Jinside = 10.** logJinside

    r = 10.0**logr
    omega = src[:, col.index("omega")]
    j_specific = r*r*omega
    # Jinside = np.cumsum(10.0**logj)
    mass = src[:, col.index("mass")]
    a = clight*j_specific/(G_cgs*mass*Msun)
    r_ISCO, j_ISCO = calculate_lso_radius_and_spin_parameter(mass, a)
    r_ISCO_nonrot, j_ISCO_nonrot = calculate_lso_radius_and_spin_parameter(mass, np.zeros(len(mass)))
    r_ISCO_maxrot, j_ISCO_maxrot = calculate_lso_radius_and_spin_parameter(mass, np.ones(len(mass)))
    # calculate dynamical timescale
    t_ff = get_local_free_fall_timescale(pfile)
    # calculate radial component of Alfven velocity
    t_alfv = get_local_alfven_timescale(pfile)

    fig = plt.figure(figsize=(8,16))
    gs = gridspec.GridSpec(150, 100)
    ax = fig.add_subplot(gs[:30, :])
    bx = fig.add_subplot(gs[30:60, :])
    cx = fig.add_subplot(gs[60:90, :])
    dx = fig.add_subplot(gs[90:150, :])

    ax.set_ylim(0,1)
    ax.set_ylabel(r"$a = cJ/GM^2$")
    bx.set_ylabel(r"$\tau_{\rm ff}$ [s]")
    bx.set_yscale('log')
    cx.set_ylabel(r"$\tau_{\rm Alfv}$ [s]")
    cx.set_ylim(1e-1, 1e13)
    cx.set_yscale('log')
    dx.set_yscale('log')
    dx.set_xlabel(r"$m \ [M_\odot]$")
    dx.plot(mass, j_ISCO_nonrot, c='#808080', ls='--', label=r"$j_{\rm ISCO}(a=0)$")
    dx.plot(mass, j_ISCO, c='k', ls='-', label=r"$j_{\rm ISCO}(a(t))$")
    dx.plot(mass, j_ISCO_maxrot, c='#808080', ls='-.', label=r"$j_\mathrm{ISCO}(a=1)$")
    dx.plot(mass, j_specific, label=r"$j=r^2\omega$")
    dx.legend(fontsize=20, loc="lower right")
    dx.set_ylabel(r"$j \ [\mathrm{cm^2\ s^{-1}}]$")
    ax.plot(mass, a)
    bx.plot(mass, t_ff)
    cx.plot(mass, t_alfv)
    fig.align_ylabels()
    if title != None:
        ax.set_title(title, fontsize=30)
    if fig_name:
        plt.savefig(fig_name)
        print(fig_name)
    else:
        plt.show()

if __name__ == "__main__":
    # tidally induced CHE
    # root = "/mnt/home/mrenzo/ceph/RUNS/CHE_jet/runs_binary_CHE/good/LOGS1/"
    # pfile = root+"/profile59.data"
    # fig_name = root+"../png/pre-CC_profiles.png"
    # summary_pfile_plot_radius(pfile, fig_name=fig_name)
    # fig_name = root+"../png/timescales.png"
    # plot_spin_tff_talfven_vs_radius(pfile, fig_name=fig_name, title="Binary CHE")
    # plot_spin_tff_talfven_vs_mass(pfile, fig_name=fig_name.replace('.png', '_mass.png'), title="Binary CHE")
    # single star CHE
    root = "/home/mrenzo/Documents/Research/Projects/CHE_jets/single_CHE_template/" # small_net/
    pfile = root+"LOGS1/CHE_single_core_collapse.data"
    fig_name = root+"pre-CC_profiles.png"
    summary_pfile_plot_radius(pfile, fig_name=fig_name)
    fig_name = root+"/png/timescales.png"
    plot_spin_tff_talfven_vs_radius(pfile, fig_name=fig_name, title="large net")
    plot_spin_tff_talfven_vs_mass(pfile, fig_name=fig_name.replace('.png', '_mass.png'), title="Single CHE")
