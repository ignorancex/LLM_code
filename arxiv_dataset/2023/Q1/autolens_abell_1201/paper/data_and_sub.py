from autoconf import conf
import autofit as af
import autolens as al
import autolens.plot as aplt

import numpy as np
import os
from os import path
import warnings

warnings.filterwarnings("ignore")

"""
__Database + Paths__
"""
workspace_path = os.getcwd()

config_path = path.join(workspace_path, "config")
conf.instance.push(new_path=config_path)

output_path = path.join(workspace_path, "output")

agg = af.Aggregator.from_database(filename="rjlens.sqlite", completed_only=True)

waveband = "f390w"
# waveband = "f814w"

plot_path = path.join(workspace_path, "paper", "images", "data_and_sub", waveband)

"""
__Grid__
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.04)


if waveband in "f390w":
    vmax_image = 0.08
else:
    vmax_image = 0.2

if waveband in "f390w":
    vmax_image_zoom = 0.04
else:
    vmax_image_zoom = 0.1


def plot_fit(
    waveband,
    search_name,
    prefix,
    bulge_cls,
    disk_cls=None,
    envelope_cls=None,
    sersic_index_range=None,
):

    agg_query = agg

    agg_query = agg_query.query(agg_query.search.unique_tag == waveband)
    agg_query = agg_query.query(agg_query.search.name == search_name)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.bulge == bulge_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.disk == disk_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.envelope == envelope_cls)

    if sersic_index_range is not None:
        agg_query = agg_query.query(
            agg_query.model.galaxies.lens.bulge.sersic_index > sersic_index_range[0]
        )
        agg_query = agg_query.query(
            agg_query.model.galaxies.lens.bulge.sersic_index < sersic_index_range[1]
        )

    include_2d = aplt.Include2D(
        critical_curves=False,
        caustics=False,
        mask=False,
        light_profile_centres=False,
        mass_profile_centres=False,
    )

    colorbar_image = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax_image, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax_image, 3), 2),
    )

    fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg_query)
    fit_imaging = list(fit_imaging_agg.max_log_likelihood_gen_from())[0]

    title_fontsize = 24

    mat_plot_2d_image = aplt.MatPlot2D(
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[-3.0, 0.0, 3.0],
            rotation="vertical",
            va="center",
        ),
        xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-3.0, 0.0, 3.0]),
        ylabel=aplt.YLabel(label=""),
        xlabel=aplt.XLabel(label=""),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
    )

    # Data #

    # mat_plot_2d = aplt.MatPlot2D(
    #     cmap=aplt.Cmap(vmax=vmax_image),
    #     title=aplt.Title(label=f"Abell 1201 ({waveband.upper()}) Image"),
    #     output=aplt.Output(
    #         filename=f"_{prefix}_data", path=plot_path, format=["png", "pdf"], format_folder=True
    #     ),
    # )
    #
    # fit_imaging_plotter = aplt.FitImagingPlotter(
    #     fit=fit_imaging, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
    # )
    #
    # fit_imaging_plotter.figures_2d(image=True)

    # Data Lens Sub #

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_image),
        title=aplt.Title(
            label=f"{waveband.upper()} Lens Subtracted Image", fontsize=title_fontsize
        ),
        colorbar=colorbar_image,
        output=aplt.Output(
            filename=f"{prefix}_data_sub",
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging, mat_plot_2d=mat_plot_2d, include_2d=include_2d
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, subtracted_image=True)

    # Data Lens Sub Zoom #

    colorbar_image = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax_image_zoom, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax_image_zoom, 3), 2),
    )

    mat_plot_2d_image = aplt.MatPlot2D(
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[-1.5, 0.0, 1.5],
            rotation="vertical",
            va="center",
        ),
        xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-1.5, 0.0, 1.5]),
        ylabel=aplt.YLabel(label=""),
        xlabel=aplt.XLabel(label=""),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
    )

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_image_zoom),
        axis=aplt.Axis(extent=[-2.0, 2.0, -2.0, 2.0]),
        title=aplt.Title(
            label=f"{waveband.upper()} Lens Subtracted Image", fontsize=title_fontsize
        ),
        colorbar=colorbar_image,
        output=aplt.Output(
            filename=f"{prefix}_data_sub_zoom",
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging, mat_plot_2d=mat_plot_2d, include_2d=include_2d
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, subtracted_image=True)


"""
Plot fits.
"""

search_name = "light[1]_light[parametric]"

if waveband in "f390w":

    plot_fit(
        waveband=waveband,
        search_name=search_name,
        bulge_cls=al.lp.EllSersic,
        disk_cls=al.lp.EllSersic,
        prefix="x2",
        sersic_index_range=[1.160, 1.162],  # f390w
    )

else:

    # plot_fit(
    #     waveband=waveband,
    #     search_name=search_name,
    #     bulge_cls=al.lp.EllSersic,
    #     disk_cls=al.lp.EllSersic,
    #     prefix="x2",
    #     sersic_index_range=[1.265, 1.270],  # f390w
    # )

    plot_fit(
        waveband=waveband,
        search_name=search_name,
        bulge_cls=al.lp.EllSersic,
        disk_cls=al.lp.EllSersic,
        envelope_cls=al.lp.EllSersic,
        prefix="x3",
    )
