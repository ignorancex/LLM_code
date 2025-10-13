#!/usr/bin/env python3

# Place import files below
import copy

import matplotlib.collections as mcol
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D

from gcgmics.common_functions import save_figures
from gcgmics.process_data import Z0Data, median_and_spread
from gcgmics.settings import Analysis, Observations, Plotting, Simulations

BASE_STYLE = {
    "markeredgecolor": "k",
    "markeredgewidth": 0.8,
}
BIRTH_STYLE = {"ls": "--"}
MW_STYLE = {**BASE_STYLE, "marker": "o", "ms": 5, "color": "#1CD6D5", "zorder": 0}
M31_STYLE = {**BASE_STYLE, "marker": "s", "ms": 6, "color": "#C3D700", "zorder": 1}
COMMON_LEGEND_PROPERTIES = {
    "labelspacing": 0.4,
    "title_fontsize": "small",
    "fontsize": "small",
}


def main():
    # Plot settings
    try:
        plt.style.use("./paper.mplstyle")
    except OSError:
        pass

    # File locations
    out_file = "fig04_dn_dlogM.pdf"
    obs_data_files = {
        "baumgardt_mw": Observations["baumgardt_2019_mw_cluster_file"],
        "caldwell_2011_m31": Observations["caldwell_2011_m31_mstar_feh_data_file"],
        "caldwell_m31_old": Observations["caldwell_2009_m31_old_data_file"],
        "caldwell_m31_int": Observations["caldwell_2009_m31_int_data_file"],
        "caldwell_m31_yng": Observations["caldwell_2009_m31_young_data_file"],
        "hunt_mw": Observations["hunt_2024_mw_NlogM_data_file"],
    }
    obs_data = {name: load_obs_data(path) for name, path in obs_data_files.items()}

    ####################################################################
    # Analysis settings
    cl = [16.0, 84.0]
    n_bins = 30
    m_bins = np.logspace(2, 8, n_bins)  # Msun / h
    dlogm = np.log10(m_bins[1:]) - np.log10(m_bins[:-1])
    mid_logmbins = 10.0 ** (np.log10(m_bins[:-1]) + dlogm / 2.0)
    data_m_bins = np.logspace(2, 8, int(n_bins / 2))  # Msun
    data_dlogm = np.log10(data_m_bins[1:]) - np.log10(data_m_bins[:-1])
    mid_data_logmbins = 10.0 ** (np.log10(data_m_bins[:-1]) + data_dlogm / 2.0)

    ####################################################################
    # Prepare observational data
    baumgardt_2019_mw_data = obs_data["baumgardt_mw"]
    caldwell_2011_m31_data = obs_data["caldwell_2011_m31"]
    caldwell_2009_m31_old_data = obs_data["caldwell_m31_old"]
    caldwell_2009_m31_int_data = obs_data["caldwell_m31_int"]
    caldwell_2009_m31_yng_data = obs_data["caldwell_m31_yng"]
    hunt_2024_mw_data = obs_data["hunt_mw"]
    hunt_dlogm = np.repeat(
        np.log10(hunt_2024_mw_data[::2, 0][1:])
        - np.log10(hunt_2024_mw_data[::2, 0][:-1]),
        [3, *[2] * (len(hunt_2024_mw_data[::2, 0][1:]) - 2), 3],
    )
    caldwell_dlogm = 0.2
    caldwell_2011_m31_mass_data = 10.0 ** caldwell_2011_m31_data[:, 1]  # Msun
    baumgardt_2019_mw_mass_data = baumgardt_2019_mw_data[:, 1]  # Msun
    caldwell_2011_dn_dlogm_m31 = (
        np.histogram(
            caldwell_2011_m31_mass_data[~np.isnan(caldwell_2011_m31_mass_data)],
            data_m_bins,
        )[0]
        / data_dlogm
    )
    baumgardt_2019_dn_dlogm_mw = (
        np.histogram(
            baumgardt_2019_mw_mass_data[~np.isnan(baumgardt_2019_mw_mass_data)],
            data_m_bins,
        )[0]
        / data_dlogm
    )
    caldwell_2009_dn_dlogm_m31_all = (
        caldwell_2009_m31_old_data[:, 1]
        + caldwell_2009_m31_int_data[:, 1]
        + caldwell_2009_m31_yng_data[:, 1]
    ) / caldwell_dlogm

    # Configuration of plotting of observational data
    obs_config = {
        "Caldwell 2009 stairs": {
            "plot_type": "stairs",
            "x": caldwell_2009_dn_dlogm_m31_all[1::2],
            "y": 10.0 ** caldwell_2009_m31_old_data[::2, 0],
            "style": create_style(
                {}, color=M31_STYLE["color"], label=r"Caldwell+(2009)", zorder=1
            ),
        },
        "Caldwell 2009 markers": {
            "plot_type": "plot",
            "x": 10.0
            ** (
                (
                    caldwell_2009_m31_old_data[::2, 0][1:]
                    + caldwell_2009_m31_old_data[::2, 0][:-1]
                )
                / 2.0
            ),
            "y": caldwell_2009_dn_dlogm_m31_all[1::2],
            "style": create_style(
                M31_STYLE, markerfacecolor="none", markevery=3, ls="none", zorder=2
            ),
        },
        "Caldwell 2011": {
            "plot_type": "plot",
            "x": mid_data_logmbins,
            "y": caldwell_2011_dn_dlogm_m31,
            "style": create_style(M31_STYLE, label=r"Caldwell+(2011)", zorder=5),
        },
        "Baumgardt 2019": {
            "plot_type": "plot",
            "x": mid_data_logmbins,
            "y": baumgardt_2019_dn_dlogm_mw,
            "style": create_style(MW_STYLE, label=r"Baumgardt+(2019)"),
        },
        "Hunt 2024 stairs": {
            "plot_type": "stairs",
            "x": hunt_2024_mw_data[:, 1] / hunt_dlogm,
            "y": np.concatenate(([0], hunt_2024_mw_data[:, 0])),
            "style": create_style({}, color="#FA200B", label=r"Hunt+(2024)", zorder=0),
        },
        "Hunt 2024 markers": {
            "plot_type": "plot",
            "x": (hunt_2024_mw_data[::2, 0][1:] + hunt_2024_mw_data[::2, 0][:-1]) / 2.0,
            "y": hunt_2024_mw_data[1::2, 1][:-1] / hunt_dlogm[1::2][:-1],
            "style": create_style(
                MW_STYLE, markerfacecolor="none", ls="none", zorder=1
            ),
        },
    }

    # Create combined Caldwell style for legend
    combined_caldwell_2009_style = create_composite_style(
        obs_config["Caldwell 2009 stairs"]["style"],
        obs_config["Caldwell 2009 markers"]["style"],
    )

    # Create combined Hunt style for legend
    combined_hunt_2024_style = create_composite_style(
        obs_config["Hunt 2024 stairs"]["style"],
        obs_config["Hunt 2024 markers"]["style"],
    )

    ####################################################################
    # Load simulation data
    all_z0_data = [Z0Data(sim) for sim in Simulations.get("Standard")["sim_list"]]
    sim_colors, sim_line_styles = [], []

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    ####################################################################
    for sim, sim_name, z0_data in zip(
        Simulations.get("Standard")["sim_list"],
        Simulations.get("Standard")["sim_names"],
        all_z0_data,
    ):
        # Cluster masses
        mcl_current = z0_data.current_cluster_m_current  # Msun
        mcl_birth = z0_data.current_cluster_m_birth  # Msun
        all_mcl_birth = z0_data.all_cluster_m_birth  # Msun

        # Calculate distribtions
        dn_dlogm_current = np.column_stack(
            [np.histogram(mass, m_bins)[0] / dlogm for mass in mcl_current.T]
        )
        dn_dlogm_birth = np.column_stack(
            [np.histogram(mass, m_bins)[0] / dlogm for mass in mcl_birth.T]
        )
        dalln_dlogm_birth = np.column_stack(
            [np.histogram(mass, m_bins)[0] / dlogm for mass in all_mcl_birth.T]
        )

        # Calculate medians and spreads
        med_dnlogm_current, spread_dnlogm_current = median_and_spread(
            dn_dlogm_current, confidence=cl
        )
        med_dnlogm_birth, _ = median_and_spread(dn_dlogm_birth, confidence=cl)
        med_dallnlogm_birth, spread_dallnlogm_birth = median_and_spread(
            dalln_dlogm_birth, confidence=cl
        )

        med_dnlogm_current[med_dnlogm_current == 0] = np.nan
        med_dnlogm_birth[med_dnlogm_birth == 0] = np.nan
        med_dallnlogm_birth[med_dallnlogm_birth == 0] = np.nan

        ################################################################
        # Plot surviving cluster mass function at z=0
        ################################################################
        # Plot median
        (line,) = ax.plot(
            mid_logmbins,
            med_dnlogm_current,
            label=sim_name,
            **Plotting["plot_styles"][sim],
        )
        # Plot scatter
        ax.fill_between(
            mid_logmbins,
            *spread_dnlogm_current,
            lw=0,
            color=line.get_color(),
            alpha=0.3,
        )

        ################################################################
        # Plot birth mass function of all clusters
        ################################################################
        err_line_dict = {"color": line.get_color(), **BIRTH_STYLE}

        ax.errorbar(
            mid_logmbins,
            med_dallnlogm_birth,
            yerr=np.abs(spread_dallnlogm_birth - med_dallnlogm_birth),
            elinewidth=2,
            capsize=3,
            **err_line_dict,
        )
        ################################################################

        sim_colors.append(line.get_color())
        sim_line_styles.append(line.get_linestyle())

    ####################################################################
    # Shade mass limit of GCs in simulations
    ax.axvspan(
        10.0 ** (3 * 0.95), Analysis["cluster_mass_limit"], alpha=0.2, color="grey"
    )

    ####################################################################
    # Plot observed surviving cluster mass functions
    ####################################################################
    # Plot median
    for _, cfg in obs_config.items():
        plot_func = getattr(ax, cfg["plot_type"])
        plot_func(cfg["x"], cfg["y"], **cfg["style"])

    ####################################################################
    # Legend settings
    ####################################################################
    # Simulation legend
    line = [[(0, 0)]]
    sim_legend_markers = []
    sim_legend_labels = []
    for sim_name, sim_color, sim_style in zip(
        Simulations.get("Standard")["sim_names"], sim_colors, sim_line_styles
    ):
        line_styles = [BIRTH_STYLE["ls"], sim_style]
        lc = mcol.LineCollection(
            len(line_styles) * line,
            linestyles=line_styles,
            colors=len(line_styles) * [sim_color],
        )
        sim_legend_markers.append(lc)
        sim_legend_labels.append(sim_name)

    sim_legend = ax.legend(
        sim_legend_markers,
        sim_legend_labels,
        markerfirst=False,
        handler_map={type(lc): HandlerDashedLines()},
        handleheight=1.25,
        labelspacing=0.0,
        loc="upper right",
    )
    for t_item, line in zip(sim_legend.get_texts(), sim_legend.get_lines()):
        t_item.set_color(line.get_color())

    # M31 legend
    m31_legend_markers = [
        plt.Line2D([0, 1], [0, 0], **combined_caldwell_2009_style),
        plt.Line2D([0, 1], [0, 0], **obs_config["Caldwell 2011"]["style"]),
    ]
    m31_legend_labels = [
        obs_config["Caldwell 2009 stairs"]["style"]["label"],
        obs_config["Caldwell 2011"]["style"]["label"],
    ]
    m31_legend = ax.legend(
        m31_legend_markers,
        m31_legend_labels,
        markerfirst=False,
        title="M31",
        loc="upper right",
        bbox_to_anchor=(1, 0.85),
        **COMMON_LEGEND_PROPERTIES,
    )

    # M31 legend
    mw_legend_markers = [
        plt.Line2D([0, 1], [0, 0], **obs_config["Baumgardt 2019"]["style"]),
        plt.Line2D([0, 1], [0, 0], **combined_hunt_2024_style),
    ]
    mw_legend_labels = [
        obs_config["Baumgardt 2019"]["style"]["label"],
        obs_config["Hunt 2024 stairs"]["style"]["label"],
    ]
    ax.legend(
        mw_legend_markers,
        mw_legend_labels,
        markerfirst=False,
        title="MW",
        loc="upper center",
        bbox_to_anchor=(0.46, 0.985),
        **COMMON_LEGEND_PROPERTIES,
    )

    ax.add_artist(sim_legend)
    ax.add_artist(m31_legend)

    ####################################################################
    # Axis settings
    ####################################################################
    ax.add_artist(sim_legend)
    ax.set(
        xlabel=r"$M_{\rm cl}\, \left[{\rm M_\odot}\right]$",
        ylabel=r"$dN\, /\, d \log M_{\rm cl}$",
        xscale="log",
        yscale="log",
        xlim=np.array([10.0 ** (3 * 0.95), 5.0e7]),
        ylim=[1.0e0, 1.0e7],
    )
    ax.tick_params(axis="x", which="major", pad=7)
    ax.minorticks_on()

    ####################################################################
    # Save figures
    save_figures(fig, out_file)

    return None


def create_style(base_style, **kwargs):
    """Create a style dictionary with deep copy and updates"""
    style = copy.deepcopy(base_style)
    style.update(kwargs)
    return style


def create_composite_style(line_style, marker_style):
    """Create composite style for legend from line and marker styles"""
    return create_style(
        line_style,
        marker=marker_style["marker"],
        markerfacecolor=marker_style["markerfacecolor"],
        ms=marker_style["ms"],
        **BASE_STYLE,
    )


def load_obs_data(file_path):
    """Load observational data from text files"""
    return np.genfromtxt(file_path, skip_header=True)


class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Establish number of lines
        numlines = len(orig_handle.get_segments())
        xdata, _ = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        leglines = []

        # Divide the vertical space where the lines will go into equal
        # parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))

        # For each line, create the line at the proper location and set
        # the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)

            # Extract properties
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]

            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]

            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]

            # Handle dashes
            if dashes[1] is None:
                legline.set_linestyle("-")
            else:
                legline.set_dashes(dashes[1])

            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)

        return leglines


if __name__ == "__main__":
    main()
