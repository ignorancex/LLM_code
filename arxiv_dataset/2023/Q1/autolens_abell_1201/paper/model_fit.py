"""
__Imports__
"""
import os
from os import path
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import autofit as af
import autolens as al
import autolens.plot as aplt

from autoconf import conf

"""
__Paths__
"""
workspace_path = os.getcwd()

config_path = path.join(workspace_path, "config")
conf.instance.push(new_path=config_path)

output_path = path.join(workspace_path, "output")

"""
___Database__

Load the database, which is already built.
"""
aggregator = af.Aggregator.from_database(filename="rjlens.sqlite", completed_only=True)

"""
__Query__
"""
waveband = "f814w"
vmax_image = 0.3
vmax_source = 0.1

# waveband = "f390w"
# vmax_image = 0.08
# vmax_source = 0.1

unique_tag = aggregator.search.unique_tag
agg_query = aggregator.query(unique_tag == waveband)

name = aggregator.search.name
agg_query = agg_query.query(
    name == "source_inversion[4]_light[fixed]_mass[total]_source[inversion]"
)

print("Total Samples Objects via `name` model query = ", len(agg_query), "\n")

"""
__Plot Fit__
"""
fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg_query)
fit_gen = fit_imaging_agg.max_log_likelihood_gen_from()

plot_path = path.join(workspace_path, "paper", "images", "model_fit")

for fit in fit_gen:

    grid_critical_curves = al.Grid2D.uniform(
        shape_native=fit.imaging.shape_native, pixel_scales=fit.imaging.pixel_scales
    )

    critical_curves = [
        fit.tracer.tangential_critical_curve_from(grid=grid_critical_curves)
    ]
    caustics = [fit.tracer.tangential_caustic_from(grid=grid_critical_curves)]

    visuals_2d = aplt.Visuals2D(critical_curves=critical_curves)

    include_2d = aplt.Include2D(
        mapper_data_pixelization_grid=False,
        mapper_source_grid_slim=False,
        mapper_source_pixelization_grid=False,
        light_profile_centres=False,
        mass_profile_centres=False,
    )

    title_fontsize = 24

    colorbar_image = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax_image, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax_image, 3), 2),
    )

    mat_plot_2d_base = aplt.MatPlot2D(
        ylabel=aplt.YLabel(label=""),
        xlabel=aplt.XLabel(label=""),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
    )

    mat_plot_2d_image = mat_plot_2d_base + aplt.MatPlot2D(
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[-3.0, 0.0, 3.0],
            rotation="vertical",
            va="center",
        ),
        xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-3.0, 0.0, 3.0]),
    )

    ### Image ###

    mat_plot_2d = mat_plot_2d_image + (
        aplt.MatPlot2D(
            title=aplt.Title(
                label=f"Image ({waveband.upper()})", fontsize=title_fontsize
            ),
            cmap=aplt.Cmap(vmin=0.0, vmax=vmax_image),
            colorbar=colorbar_image,
            output=aplt.Output(
                path=path.join(plot_path, waveband),
                format=["png", "pdf"],
                format_folder=True,
                bbox_inches="tight",
            ),
        )
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit, mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
    )
    fit_imaging_plotter.figures_2d(image=True)

    ## Model Image ##

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Model Image ({waveband.upper()})", fontsize=title_fontsize
        ),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_image),
        colorbar=colorbar_image,
        output=aplt.Output(
            path=path.join(plot_path, waveband),
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )
    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit, mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
    )
    fit_imaging_plotter.figures_2d(model_image=True)

    ## Lensed Source Model ##

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Model Lensed Source ({waveband.upper()})", fontsize=title_fontsize
        ),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_source),
        colorbar=colorbar_source,
        output=aplt.Output(
            path=path.join(plot_path, waveband),
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )
    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit, mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, model_image=True)

    visuals_2d = aplt.Visuals2D(caustics=caustics)

    ## Source Reconstruction ##

    colorbar_source = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax_source, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax_source, 3), 2),
    )

    mat_plot_2d_source = mat_plot_2d_base + aplt.MatPlot2D(
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[-1.5, 0.0, 1.5],
            rotation="vertical",
            va="center",
        ),
        xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-1.5, 0.0, 1.5]),
    )

    mat_plot_2d = mat_plot_2d_source + aplt.MatPlot2D(
        axis=aplt.Axis(extent=[-1.8, 1.8, -1.8, 1.8]),
        title=aplt.Title(
            label=f"Reconstructed Source ({waveband.upper()})", fontsize=title_fontsize
        ),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_source),
        colorbar=colorbar_source,
        voronoi_drawer=aplt.VoronoiDrawer(edgecolor=None),
        output=aplt.Output(
            path=path.join(plot_path, waveband),
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )
    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit, mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)
