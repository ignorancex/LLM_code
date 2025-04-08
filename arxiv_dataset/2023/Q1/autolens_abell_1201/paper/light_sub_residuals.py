import numpy as np
from matplotlib.patches import Circle
import os
from os import path
import warnings

from autoconf import conf
import autofit as af
import autolens as al
import autolens.plot as aplt

warnings.filterwarnings("ignore")

"""
__Database + Paths__
"""
workspace_path = os.getcwd()

config_path = path.join(workspace_path, "config")
conf.instance.push(new_path=config_path)

output_path = path.join(workspace_path, "output")

agg = af.Aggregator.from_database(filename="rjlens.sqlite", completed_only=False)

waveband = "f390w"
# waveband = "f814w"

plot_path = path.join(
    workspace_path, "paper", "images", "light_sub_residuals", waveband
)

if waveband in "f390w":
    vmax = 0.08
else:
    vmax = 0.2

vmax_norm = 2.5
vmin_norm = -2.5

grid = al.Grid2D.uniform(shape_native=(400, 400), pixel_scales=0.04)


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

    fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg_query)
    fit_imaging = list(fit_imaging_agg.max_log_likelihood_gen_from())[0]

    include_2d = aplt.Include2D(
        critical_curves=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        mapper_data_pixelization_grid=False,
    )

    patch_overlay = aplt.PatchOverlay(edgecolor="cy", alpha=0.7)

    title_fontsize = 24

    colorbar_image = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax, 3), 2),
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

    ### Data ###

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        title=aplt.Title(label=f"Data ({waveband.upper()})", fontsize=title_fontsize),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax),
        colorbar=colorbar_image,
        output=aplt.Output(
            filename=f"{prefix}_data",
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging, mat_plot_2d=mat_plot_2d, include_2d=include_2d
    )
    fit_imaging_plotter.figures_2d(image=True)

    visuals_2d = aplt.Visuals2D(
        critical_curves=[fit_imaging.tracer.critical_curves_from(grid=grid)[0]]
    )

    include_2d = aplt.Include2D(
        critical_curves=True,
        light_profile_centres=False,
        mass_profile_centres=False,
        mapper_data_pixelization_grid=False,
    )

    ### Model Image ###

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Model Image ({waveband.upper()})", fontsize=title_fontsize
        ),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax),
        colorbar=colorbar_image,
        output=aplt.Output(
            filename=f"{prefix}_image",
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging,
        mat_plot_2d=mat_plot_2d,
        include_2d=include_2d,
        visuals_2d=visuals_2d,
    )
    fit_imaging_plotter.figures_2d(model_image=True)

    ### Normalized Residual Map ###

    colorbar_norm = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(vmin_norm, vmax_norm, 3), 2),
        manual_tick_labels=np.round(np.linspace(vmin_norm, vmax_norm, 3), 2),
    )

    visuals_2d = aplt.Visuals2D(
        critical_curves=[fit_imaging.tracer.critical_curves_from(grid=grid)[0]],
        patches=[Circle(xy=(0.0, 0.0), radius=0.3, fill=True)],
    )

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        patch_overlay=patch_overlay,
        title=aplt.Title(
            label=f"Residuals ({waveband.upper()})", fontsize=title_fontsize
        ),
        cmap=aplt.Cmap(vmin=vmin_norm, vmax=vmax_norm),
        colorbar=colorbar_norm,
        output=aplt.Output(
            filename=f"{prefix}_norm",
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging,
        mat_plot_2d=mat_plot_2d,
        include_2d=include_2d,
        visuals_2d=visuals_2d,
    )
    fit_imaging_plotter.figures_2d(normalized_residual_map=True)

    ### Normalized Residual Map (Zoomed) ###

    mat_plot_2d_zoom = mat_plot_2d_base + aplt.MatPlot2D(
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[-0.9, 0.0, 0.9],
            rotation="vertical",
            va="center",
        ),
        xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-0.9, 0.0, 0.9]),
    )

    mat_plot_2d = mat_plot_2d_zoom + aplt.MatPlot2D(
        patch_overlay=patch_overlay,
        title=aplt.Title(
            label=f"Residuals Zoom ({waveband.upper()})", fontsize=title_fontsize
        ),
        axis=aplt.Axis(extent=[-1.0, 1.0, -1.0, 1.0]),
        cmap=aplt.Cmap(vmin=vmin_norm, vmax=vmax_norm),
        colorbar=colorbar_norm,
        output=aplt.Output(
            filename=f"{prefix}_norm_zoomed",
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging,
        mat_plot_2d=mat_plot_2d,
        include_2d=include_2d,
        visuals_2d=visuals_2d,
    )
    fit_imaging_plotter.figures_2d(normalized_residual_map=True)


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
