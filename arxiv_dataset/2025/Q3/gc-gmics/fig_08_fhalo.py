#!/usr/bin/env python3
# Place import files below
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from gcgmics.common_functions import save_figures
from gcgmics.process_data import Z0Data, median_and_spread
from gcgmics.settings import Analysis, Observations, Plotting, Simulations


def main():
    # Plot settings
    try:
        plt.style.use("./paper.mplstyle")
    except OSError:
        pass

    # File locations
    out_file = "fig08_mcfield_mfield.pdf"
    out_file_template = "{}_vs_tlb.pdf"
    horta_mw_data = np.genfromtxt(
        Observations["horta_2021_zetagc_file"], skip_header=True
    )

    # Analysis settings
    n_bins = 16
    cl = [16.0, 84.0]
    plt.rc("hatch", color="k", linewidth=0.5)
    common_fill_style = {
        "hatch": "xxx",
        "color": "None",
        "linewidth": 0.5,
        "alpha": 0.3,
    }

    # Load data for figures
    all_z0_data = [Z0Data(sim) for sim in Simulations.get("Standard")["sim_list"]]
    property_list = ["fhalo_cl", "fhalo_gc"]

    # Select only Enhanced and Suppressed simulations for plotting
    selected_sim_list = np.asarray(Simulations.get("Standard")["sim_list"])[[0, 2]]
    selected_sim_names = np.asarray(Simulations.get("Standard")["sim_names"])[[0, 2]]
    selected_z0_data = np.asarray(all_z0_data)[[0, 2]]

    ####################################################################
    # Create figures
    fig, axs = plt.subplots(
        len(property_list),
        1,
        figsize=(8, 8 * (1.0 + 0.3 * (len(property_list) - 1.0))),
        sharex=True,
        gridspec_kw={
            "width_ratios": [1],
            "height_ratios": [1.0 / len(property_list) for _ in property_list],
            "wspace": 0,
            "hspace": 0,
        },
    )

    if Plotting["individual_panels"]:
        indiv_figs = [plt.figure(figsize=(8, 8)) for _ in property_list]
        indiv_axs = [fig.add_subplot(111) for fig in indiv_figs]
    else:
        indiv_figs = [None] * len(property_list)
        indiv_axs = [None] * len(property_list)

    ####################################################################
    # Iterate over each simulation and plot data
    for sim, sim_name, z0_data in zip(
        selected_sim_list, selected_sim_names, selected_z0_data
    ):
        # Reina-Campos definition
        rc_cl, rc_gcs, _, _ = z0_data.reina_campos_f_halo(
            outer_radius=Analysis["outer_radius"], m_gc=Analysis["gc_mass"]
        )

        # Radial calculation
        (radius_mid_bins, f_halo_cl, f_halo_gcs, f_halo_init_cl, f_halo_init_gcs) = (
            z0_data.f_halo_vs_distance(
                n_bins=n_bins,
                outer_radius=Analysis["outer_radius"],
                m_gc=Analysis["gc_mass"],
            )
        )
        # max_rhalf = z0_data.max_rhalf

        # FHalo in radial bins
        med_fcl, spread_fcl = median_and_spread(f_halo_cl, confidence=cl)
        med_fgc, spread_fgc = median_and_spread(f_halo_gcs, confidence=cl)
        (med_fcl_init, _) = median_and_spread(f_halo_init_cl, confidence=cl)
        (med_fgc_init, _) = median_and_spread(f_halo_init_gcs, confidence=cl)

        # FHalo (Reina-Campos definition) data
        med_rc_cl, spread_rc_cl = median_and_spread(rc_cl, confidence=cl)
        med_rc_gc, spread_rc_gc = median_and_spread(rc_gcs, confidence=cl)

        ################################################################
        # Plot lines and fills
        for ax_obj in [axs, indiv_axs]:
            if ax_obj[0] is not None:
                # Plot median
                (line,) = ax_obj[0].plot(
                    radius_mid_bins,
                    med_fcl,
                    label=sim_name,
                    **Plotting["plot_styles"][sim],
                )
                # Plot scatter
                new_rgba = [*matplotlib.colors.to_rgba(line.get_color())]
                common_fill_style.update({"edgecolor": new_rgba})
                ax_obj[0].fill_between(
                    radius_mid_bins, *spread_fcl, **common_fill_style
                )

                # Plot evolved initial mass
                # Median
                ax_obj[0].plot(
                    radius_mid_bins, med_fcl_init, ls="--", color=line.get_color()
                )

                # Plot FHalo (Reina-Campos definition)
                ax_obj[0].axhline(med_rc_cl, ls=":", color=line.get_color(), zorder=1)
                ax_obj[0].axhspan(
                    *spread_rc_cl, color=line.get_color(), zorder=0, ec=None, alpha=0.2
                )
                # Plot median
                (line,) = ax_obj[1].plot(
                    radius_mid_bins,
                    med_fgc,
                    label=sim_name,
                    **Plotting["plot_styles"][sim],
                )
                fill_style_dict = {**common_fill_style}
                fill_style_dict.update({"edgecolor": line.get_color()})
                # Plot scatter
                ax_obj[1].fill_between(
                    radius_mid_bins,
                    *spread_fgc,
                    **common_fill_style,
                )

                # Plot evolved initial mass
                # Median
                ax_obj[1].plot(
                    radius_mid_bins, med_fgc_init, ls="--", color=line.get_color()
                )

                # Plot FHalo (Reina-Campos definition)
                ax_obj[1].axhline(med_rc_gc, ls=":", color=line.get_color(), zorder=1)
                ax_obj[1].axhspan(
                    *spread_rc_gc, color=line.get_color(), zorder=0, ec=None, alpha=0.2
                )

    ####################################################################
    # Plot Horta+2021 MW data
    for ax in np.concatenate((axs, indiv_axs)):
        if ax is not None:
            # Plot median
            (line,) = ax.plot(
                horta_mw_data[:, 0],
                1.5 * horta_mw_data[:, 1] * 0.01,
                color="grey",
                marker="*",
                markerfacecolor="yellow",
                markeredgecolor="yellow",
                markevery=8,
            )
            # Plot scatter
            ax.fill_between(
                horta_mw_data[:, 0],
                *horta_mw_data[:, 2:].T * 1.5 * 0.01,
                color=line.get_color(),
                edgecolor=line.get_color(),
                linewidth=0.5,
                alpha=0.3,
            )

    ####################################################################
    # Legend
    legend_markers = [
        plt.Line2D([0, 1], [0, 0], color="k", linestyle="--"),
        plt.Line2D([0, 1], [0, 0], color="k", linestyle="-"),
        plt.Line2D([0, 1], [0, 0], color="k", linestyle=":"),
        (
            patches.Patch(color="k", alpha=0.3, linewidth=0),
            plt.Line2D(
                [0, 1],
                [0, 0],
                color="grey",
                linestyle="-",
                marker="*",
                markerfacecolor="yellow",
                markeredgecolor="yellow",
            ),
        ),
    ]
    legend_labels = [
        "Fully disrupted",
        "Partially disrupted",
        r"$F^{\rm halo}$",
        "Horta+(2021b)",
    ]
    orig_legend_settings = {
        "markerfirst": False,
        "loc": "upper right",
        "handlelength": 0.0,
        "handletextpad": 0.0,
        "labelspacing": 0.0,
    }
    new_legend_settings = {
        "handles": legend_markers,
        "labels": legend_labels,
        "bbox_to_anchor": (0.0, 1.02, 1.0, 0.102),
        "loc": "lower left",
        "ncols": 2,
        "mode": "expand",
        "borderaxespad": 0.0,
        "frameon": True,
        "fancybox": True,
        "framealpha": 0.75,
    }

    for ax in axs:
        ax.minorticks_on()
        ax.set(yscale="log")
    axs[0].set(
        ylabel=r"$\zeta_{\rm CL} = M^{\rm halo}_{\rm CL}\, /\, M^{\rm halo}_{\rm tot}$",
        ylim=[None, 2.5e-1],
    )
    axs[1].set(
        xlabel=r"$r\, \left[{\rm kpc}\right]$",
        xlim=[0.0, None],
        ylabel=r"$\zeta_{\rm GC} = M^{\rm halo}_{\rm GC}\, /\, M^{\rm halo}_{\rm tot}$",
        ylim=[None, 8.0e-2],
    )
    orig_legend = axs[0].legend(**orig_legend_settings)
    for t_item, line in zip(orig_legend.get_texts(), orig_legend.get_lines()):
        t_item.set_color(line.get_color())
    axs[0].legend(**new_legend_settings)
    axs[0].add_artist(orig_legend)

    ####################################################################
    # Individual figures
    if Plotting["individual_panels"]:
        for indiv_ax in indiv_axs:
            indiv_ax.minorticks_on()
            ax0_orig_legend = indiv_ax.legend(**orig_legend_settings)
            for t_item, line in zip(
                ax0_orig_legend.get_texts(), ax0_orig_legend.get_lines()
            ):
                t_item.set_color(line.get_color())
            indiv_ax.legend(**new_legend_settings)
            indiv_ax.add_artist(ax0_orig_legend)
            indiv_ax.set(
                xlabel=r"$r\, \left[{\rm kpc}\right]$", xlim=[0.0, None], yscale="log"
            )
        indiv_axs[0].set(
            ylabel=r"$\zeta_{\rm CL} = M_{\rm field,\, CL}\, /\, M_{\rm field,\, tot}$",
            ylim=[None, 2.5e-1],
        )
        indiv_axs[1].set(
            ylabel=r"$\zeta_{\rm GC} = M_{\rm field,\, GC}\, /\, M_{\rm field,\, tot}$",
            ylim=[None, 8.0e-2],
        )

    ####################################################################
    # Save figures
    print(f"Writing {out_file}")
    save_figures(fig, out_file)

    if Plotting["individual_panels"]:
        for prop_name, fig in zip(property_list, indiv_figs):
            print(f"Writing {out_file_template.format(prop_name)}")
            save_figures(fig, out_file_template.format(prop_name))

    return None


if __name__ == "__main__":

    main()
