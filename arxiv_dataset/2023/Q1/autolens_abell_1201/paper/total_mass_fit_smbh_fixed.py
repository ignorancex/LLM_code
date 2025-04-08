import numpy as np
import os
from os import path
from autoconf import conf
import autofit as af
import autolens as al
import autolens.plot as aplt
import warnings

warnings.filterwarnings("ignore")

"""
__Database + Paths__
"""
workspace_path = os.getcwd()

config_path = path.join(workspace_path, "config")
conf.instance.push(new_path=config_path)

output_path = path.join(workspace_path, "output")

agg = af.Aggregator.from_database(
    filename="rjlens_no_lens_light_fixed.sqlite", completed_only=False
)

waveband = "f390w"
# waveband = "f814w"

plot_path = path.join(
    workspace_path, "paper", "images", "total_mass_fit_smbh_fixed", waveband
)

vmax = 1.5
vmin = -1.5

if waveband in "f390w":
    vmax_zoom = 0.02
else:
    vmax_zoom = 0.06


def plot_fit(waveband, search_name, mass_label):

    einstein_radius = 0.9

    agg_query = agg

    agg_query = agg_query.query(agg_query.search.unique_tag == waveband)
    agg_query = agg_query.query(agg_query.search.name == search_name)
    agg_query = agg_query.query(
        agg_query.model.galaxies.lens.smbh.einstein_radius == einstein_radius
    )

    fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg_query)
    fit_imaging = list(fit_imaging_agg.max_log_likelihood_gen_from())[0]

    include_2d = aplt.Include2D(
        critical_curves=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        mapper_data_pixelization_grid=False,
    )

    title_fontsize = 24

    ### Lensed Source Reconstruction (Zoomed) ###

    colorbar_image = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax_zoom, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax_zoom, 3), 2),
    )

    mat_plot_2d_base = aplt.MatPlot2D(
        ylabel=aplt.YLabel(label=""),
        xlabel=aplt.XLabel(label=""),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
    )

    mat_plot_2d_image_zoom = mat_plot_2d_base + aplt.MatPlot2D(
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[-1.1, -0.45, 0.2],
            rotation="vertical",
            va="center",
        ),
        xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-0.9, -0.25, 0.4]),
    )
    mat_plot_2d = mat_plot_2d_image_zoom + aplt.MatPlot2D(
        title=aplt.Title(
            label=rf"{mass_label} w/ $10^{{11}} M_{{\odot}}$ SMBH ({waveband.upper()})",
            fontsize=title_fontsize,
        ),
        axis=aplt.Axis(extent=[-1.0, 0.5, -1.2, 0.3]),
        colorbar=colorbar_image,
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_zoom),
        output=aplt.Output(
            filename=f"{einstein_radius}_image_zoomed",
            path=plot_path,
            format=["png", "pdf"],
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging, mat_plot_2d=mat_plot_2d, include_2d=include_2d
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, model_image=True)


"""
Make Mass Profile Plotters.
"""

search_name = "mass_total[1]_mass[total]_source_smbh_fixed"

mass_profile_plotter = plot_fit(
    waveband=waveband, search_name=search_name, mass_label="BPL"
)
