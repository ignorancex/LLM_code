import os
import sys
import numpy as np
import astropy.io.ascii as asc
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import interp1d
import blueshift_util
from scipy.signal import savgol_filter


def reshape_and_mask(var, astropy_table, shape=(100, 100)):
    mask = (astropy_table["inwind"] < 0)
    x = astropy_table["x"].reshape(shape)
    z = astropy_table["z"].reshape(shape)
   #astropy_table[var] = astropy_table[var][(astropy_table["inwind"]>=0)]= 0.0
    var = np.ma.masked_array(astropy_table[var].reshape(shape), mask=mask)
    #var[astropy_table["inwind"]<0] = 0.0
    return (x, z, var)


def invisible_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_continuum_normalised(ax, s, colname="A05P0.50", vel_max=10, color="C0", mappable=None, **plot_kwargs):

    wavelength = s["Lambda"]
    velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs
    velocity = velocity.to(u.km/u.s)
    fcont = blueshift_util.fit_continuum(wavelength, s[colname])

    fcont = blueshift_util.fit_continuum(wavelength, s[colname])
    flux = savgol_filter(s[colname]/fcont, 21, 3)

    x = velocity.cgs.value/1e8
    #plt.scatter(x, flux)
    select = (np.fabs(x) <= (1.1*vel_max))

    x = x[select]
    y = flux[select]
    interp_func = interp1d(x, y)
    x = np.linspace(-1.09*vel_max, 1.09*vel_max, 1000)
    y = interp_func(x)

    blueshift_util.gradient_fill(
        x, y, ax=ax, mappable=mappable, alpha=1.0, fill_color=color, **plot_kwargs)
    ax.set_xlim(-vel_max, vel_max)
    ax.set_ylim(1, 3.5)


def make_plot(var, data, title, cmap="viridis", log=True, four=True, signs=[1, 1, 1, 1], vmin=0, vmax=1):
    x, z, var = reshape_and_mask(var, data)
    if log:
        q = np.log10(var)
        title = "log " + title
    else:
        q = var

    plt.pcolormesh(x, z, signs[0]*q, cmap=cmap, vmin=vmin, vmax=vmax)
    if four:
        plt.pcolormesh(-x, z, signs[1] * q, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.pcolormesh(-x, -z, signs[2] * q, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.pcolormesh(x, -z, signs[3] * q, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xlim(-1e18, 1e18)
        plt.ylim(-1e18, 1e18)
    else:
        plt.xlim(0, 1e18)
        plt.ylim(0, 1e18)
    plt.title("{}".format(title), fontsize=16)
    plt.colorbar()
    # plt.loglog()


def make_plot_observer(var, data, title, cmap="viridis", log=True, four=True, signs=[1, 1, 1, 1], vmin=0, vmax=1):
    x, z, var = reshape_and_mask(var, data)
    if log:
        q = np.log10(var)
        title = "log " + title
    else:
        q = -var

    angle = np.radians(75)
    project1 = np.cos(np.arctan2(z, x) - angle)
    project2 = np.cos(np.arctan2(z, -x) - angle)
    project3 = np.cos(np.arctan2(-z, -x) - angle)
    project4 = np.cos(np.arctan2(-z, x) - angle)

    shading = "flat"
    kwargs = {"cmap": cmap, "vmin": vmin, "vmax": vmax,
              "antialiased": False, "linewidth": 0.0}
    mappable = plt.pcolormesh(x, z, project1 * q, **kwargs)
    if four:
        plt.pcolormesh(-x, z, project2 * q, **kwargs)
        plt.pcolormesh(-x, -z, project3 * q, **kwargs)
        plt.pcolormesh(x, -z, project4 * q, **kwargs)
        plt.xlim(-1e18, 1e18)
        plt.ylim(-1e18, 1e18)
    else:
        plt.xlim(0, 1e18)
        plt.ylim(0, 1e18)
    # plt.gca().axvline(0)
    #plt.title("{}".format(title), fontsize=16)
    cax = plt.gcf().add_axes([0.227, 0.15, 0.2, 0.05])
    cbar = plt.colorbar(orientation="horizontal", cax=cax)
    cbar.set_label(r"Projected Velocity ($10^3$~km~s$^{-1}$)")
    # plt.loglog()
    return mappable


def make_figure(alpha=1):
    print("Making figure 2...", end="")

    #alpha = 0.5
    data_dir = blueshift_util.get_data_dir_for_alpha(alpha=alpha)

    if alpha == 0.5:
        f1 = "{}/run47_diskmax_3e16".format(data_dir)
        f2 = "{}/run47_thmin45_rv1e19_vinf1p0_mdotw5p0".format(data_dir)
        vel_max = 7
    elif alpha == 1:
        f1 = "{}/run128_diskmax_3e16".format(data_dir)
        #f1 = "{}/old_spectra_20cycles/run128_thmin45_rv1e19_vinf1p0_mdotw5p0".format(data_dir)
        f2 = "{}/run128_thmin45_rv1e19_vinf1p0_mdotw5p0".format(data_dir)
        vel_max = 4

    root_fname = f2
    plotit = True

    data = asc.read("{}.master.txt".format(root_fname))
    data["x"] = data["xcen"]
    data["z"] = data["zcen"]
    # spec1 = asc.read("{}.spec".format(root_fname))
    spec2 = asc.read("{}.spec".format(f1))
    spec1 = asc.read("{}.spec".format(f2))

    angle = blueshift_util.fiducial_angle

    plt.figure(figsize=(8, 5.2))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)

    data["v_z"] /= 3e10
    vz = data["v_z"] / 1e8
    vx = data["v_x"] / 1e8
    vr = np.sqrt(vx * vx + vz * vz)
    data["vr"] = vr

    mappable = make_plot_observer("vr", data, "v_z/c", log=False,
                                  cmap="RdBu_r", signs=[1, 1, -1, -1], vmin=-vel_max, vmax=vel_max)
    ax1.plot([-1e18, 1e18], [0, 0], c="k", alpha=0.6, lw=2)

    r = np.logspace(0, 18, 10000)
    theta = np.radians(90-angle)
    rcostheta = r * np.cos(theta)
    rsintheta = r * np.sin(theta)
    ax1.plot(rcostheta, rsintheta, ls=(0, (1, 1)), c="k", lw=3)

    invisible_axis(ax1)
    axin1 = ax1.inset_axes([0.545, 0.885, 0.15, 0.15])
    invisible_axis(axin1)

    # read the eye image!
    import matplotlib.image as mpimg
    axin1.imshow(mpimg.imread('{}/eye2.png'.format(data_dir)))

    # now plot the line profile
    ax2 = plt.subplot2grid((8, 3), (2, 2), rowspan=4)
    plot_continuum_normalised(ax2, spec2, colname="A{:02d}P0.50".format(
        angle), vel_max=vel_max, mappable=mappable, label="Transparent", lw=4, color="C6")
    plot_continuum_normalised(ax2, spec1, colname="A{:02d}P0.50".format(
        angle), vel_max=vel_max, mappable=mappable, label="Opaque", lw=4, color="C0")

    # vline_max =
    vline_max = 2.45
    if alpha == 1:
        vline_max = 2.85
    ax2.vlines([0], 1, vline_max, color="k", lw=2, ls=(0, (5, 1)), alpha=0.7)
    ax2.set_title(r"C{\textsc{iv}~$\lambda$1550")
    ax2.set_xlabel(r"Velocity ($10^3$~km~s$^{-1}$)")
    ax2.set_ylabel(r"$F_\lambda/F_0$", labelpad=-1)
    ax2.legend(frameon=False, loc=2, borderpad=0.1, labelspacing=0.25)
    plt.subplots_adjust(left=0.0, right=0.99, top=1,
                        bottom=0.0, hspace=0.2, wspace=0.27)
    figure_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'Figures'))
    plt.savefig("{}/fig2.pdf".format(figure_dir), dpi=200)
    print("Done.")


if __name__ == "__main__":
    blueshift_util.set_plot_defaults()
    
    alpha = 1
    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])
    make_figure(alpha=alpha)
