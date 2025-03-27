from .colorline import colorline
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors, cm


def plot_cmd(bp_rp, mg, title=None, ax=None, cmap="Greys", bins=500):
    """
        plot_cmd(bp_rp, mg, title=None, ax=None, cmap="Greys", bins=500)

        Plot a two-dimensional histogram of the Gaia color-magnitude diagram (CMD).

        Parameters
        ----------
        bp_rp : array_like
            Gaia Gbp-Grp color.
        mg : array_like
            Gaia G-band absolute magnitude (same size as `bp_rp`).
        title : str, optional
            Plot title (default: None).
        ax : Axes object, optional
            Matplotlib Axes in which to plot (default: None).
        cmap : str, optional
            Colormap (default: "Greys").
        bins : int, optional
            Number of bins in the 2D histogram (default: 500).

        Returns
        -------
        ax : Axes object, optional
            Matplotlib Axes.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist2d(bp_rp, mg, bins=bins, cmap=cmap, norm=colors.PowerNorm(0.25), zorder=0.5)
    if ~ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")
    if title is not None:
        plt.title(title)
    plt.show()

    return ax


def plot_track(bp_rp, mg, title=None, ax=None, c=None, label=None, **kwargs):
    """
        plot_track(bp_rp, mg, title=None, ax=None, c=None, label=None, **kwargs)

        Plot evolutionary track over the Gaia color-magnitude diagram (CMD).

        Parameters
        ----------
        bp_rp : array_like
            Gaia Gbp-Grp color.
        mg : array_like
            Gaia G-band absolute magnitude (same size as `bp_rp`).
        title : str, optional
            Plot title (default: None).
        ax : Axes object, optional
            Matplotlib Axes in which to plot (default: None).
        c : str, optional
            Plot color (default: None).
        label : str, optional
            Plot legend label (default: None).
        **kwargs : optional
            Additional arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : Axes object, optional
            Matplotlib Axes.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if c is None:
        ax.plot(bp_rp, mg, label=label, **kwargs)
    else:
        ax.plot(bp_rp, mg, c, label=label, **kwargs)
    if ~ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")
    if title is not None:
        plt.title(title)
    plt.show()

    return ax


def plot_combined_isomasses(tracks, mass=np.arange(0.1, 1.2, 0.1), mass_res=0.007, ax=None,
                            is_colorline=True, c='k', plot_pre_ms=True, print_mass=True,
                            mass_label_x_offset=-0.1, mass_label_y_offset=0.25, **kwargs):
    """
        plot_combined_isomasses(tracks, mass=np.arange(0.1, 1.2, 0.1), mass_res=0.007, ax=None,
                                is_colorline=True, c='k', print_mass=True, plot_pre_ms=True, **kwargs)

        Plot evolutionary track over the Gaia color-magnitude diagram (CMD).

        Parameters
        ----------
        tracks : Table
            Stellar evolution tracks, generated using `stam.gentracks.get_combined_isomasses` (a Table with columns `mass`, `bp_rp`, `mg`, and `mh`).
        mass : array_like, optional
            Stellar track mass array, in Msun (default: `np.arange(0.1, 1.2, 0.1)` Msun).
        mass_res : float, optional
            Mass resolution, in Msun (default: 0.007 Msun).
        ax : Axes object, optional
            Matplotlib Axes in which to plot (default: None).
        is_colorline : bool, optional
            Plot tracks in gradient colors by mass and metallicity? (default: True)
        c : str, optional
            Tracks plot color, if `is_colorline=False` (default: 'k').
        plot_pre_ms : bool, optional
            Plot the pre-main-sequence part of the tracks? (default: True)
        print_mass : bool, optional
            Print mass label next to tracks? (default: True).
        mass_label_x_offset : float, optional
            Mass label x-axis offset (default: -0.1).
        mass_label_y_offset : float, optional
            Mass label y-axis offset (default: 0.25).
        **kwargs : optional
            Additional arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : Axes object, optional
            Matplotlib Axes.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if is_colorline:
        if not (ax.lines or ax.collections):
            # nothing plotted yet, need to set axes limits for LineCollection (otherwise it's set to [0,1])
            set_lim = True
            xmin = 0
            xmax = 1
            ymin = 0
            ymax = 1
        else:
            set_lim = False

        # get the global color limits:
        ms_idx = np.array(tracks["stage"] == 1)
        # main sequence
        mh_min = np.min(tracks["mh"][ms_idx])
        mh_max = np.max(tracks["mh"][ms_idx])
        # pre main sequence
        age_min = 1e3 * np.min(tracks["age"][~ms_idx])
        age_max = 1e3 * np.max(tracks["age"][~ms_idx])

    for m in mass:
        idx = np.abs(tracks["mass"] - m) <= mass_res
        bp_rp = np.array(tracks["bp_rp"])[idx]
        mg = np.array(tracks["mg"])[idx]
        mh = np.array(tracks["mh"])[idx]
        age = np.array(tracks["age"])[idx]
        ms_idx = np.array(tracks["stage"] == 1)[idx]

        if is_colorline:
            # main sequence
            if np.any(ms_idx):
                lc_ms = colorline(bp_rp[ms_idx], mg[ms_idx], mh[ms_idx], cmap=cm.autumn,
                                  norm=colors.PowerNorm(0.5, vmin=mh_min, vmax=mh_max), ax=ax)
            # pre main sequence
            if np.any(~ms_idx):
                lc = colorline(bp_rp[~ms_idx], mg[~ms_idx], 1e3 * age[~ms_idx], cmap=cm.summer,
                               norm=colors.PowerNorm(0.5, vmin=age_min, vmax=age_max), ax=ax)

            if set_lim:
                xmin = np.minimum(xmin, np.min(bp_rp))
                xmax = np.maximum(xmax, np.max(bp_rp))
                ymin = np.minimum(ymin, np.min(mg))
                ymax = np.maximum(ymax, np.max(mg))
        else:
            ax.plot(bp_rp[ms_idx], mg[ms_idx], color=c, linewidth=2, **kwargs)
            if plot_pre_ms:
                ax.plot(bp_rp[~ms_idx], mg[~ms_idx], color=c, linewidth=2, **kwargs)

        if print_mass:
            coef = np.polyfit(bp_rp[0:25], mg[0:25], 1)
            p = np.poly1d(coef)
            text_x = bp_rp[0] + mass_label_x_offset
            text_y = p(text_x) + mass_label_y_offset
            text_position = np.array([text_x, text_y])

            text_angle = ax.transData.transform_angles(np.array((np.rad2deg(np.arctan(coef[0])),)),
                                                       text_position.reshape((1, 2)))[0]

            ax.text(text_position[0], text_position[1], f"{m:.3} $M_{{\odot}}$",
                    fontsize=14, color='k', horizontalalignment="right",
                    rotation=text_angle, rotation_mode='anchor', clip_on=True)

    if is_colorline:
        fig = plt.gcf()
        # cax1 = fig.add_axes([0.1, 0.1, 0.03, 0.8])
        if np.any(ms_idx):
            plt.colorbar(cm.ScalarMappable(norm=colors.PowerNorm(0.5, vmin=mh_min, vmax=mh_max),
                                           cmap=cm.autumn), label="[M/H]")

        if np.any(~ms_idx):
            cax2 = fig.add_axes([0.12, 0.9, 0.62, 0.03])
            cb2 = plt.colorbar(cm.ScalarMappable(norm=colors.PowerNorm(0.5, vmin=age_min, vmax=age_max),
                                                 cmap=cm.summer), cax=cax2, label="Age [Myr]", orientation="horizontal")
            cb2.ax.xaxis.set_ticks_position('top')
            cb2.ax.xaxis.set_label_position('top')

        if set_lim:
            if print_mass:
                ax.set_xlim(0.7*xmin, 1.1*xmax)
            else:
                ax.set_xlim(0.9*xmin, 1.1*xmax)
            ax.set_ylim(0.99*ymin, 1.01*ymax)

    if ~ax.yaxis_inverted():
        ax.invert_yaxis()

    plt.show()

    return ax
