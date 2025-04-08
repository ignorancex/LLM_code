import numpy as np
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

agg = af.Aggregator.from_database(
    filename="rjlens_no_lens_light.sqlite", completed_only=False
)

# waveband = "f390w"
waveband = "f814w"

if waveband in "f390w":
    vmax_zoom = 0.02
else:
    vmax_zoom = 0.06

if waveband in "f390w":
    vmax_no_zoom = 0.07
else:
    vmax_no_zoom = 0.15

plot_path = path.join(workspace_path, "paper", "images", "total_mass_fit", waveband)

"""
__Grid__
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.04)

vmax = 2.5
vmin = -2.5


def plot_fit(
    waveband,
    search_name,
    mass_cls,
    mass_tag,
    mass_label,
    smbh_cls=None,
    plot_both_crits=True,
):

    agg_query = agg

    agg_query = agg_query.query(agg_query.search.unique_tag == waveband)
    agg_query = agg_query.query(agg_query.search.name == search_name)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.mass == mass_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.smbh == smbh_cls)

    fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg_query)
    fit_imaging = list(fit_imaging_agg.max_log_likelihood_gen_from())[0]

    grid_critical_curves = al.Grid2D.uniform(
        shape_native=fit_imaging.imaging.shape_native, pixel_scales=0.04
    )

    if plot_both_crits:

        critical_curves = fit_imaging.tracer.critical_curves_from(
            grid=grid_critical_curves
        )

    else:

        critical_curves = [
            fit_imaging.tracer.tangential_critical_curve_from(grid=grid_critical_curves)
        ]

    visuals_2d = aplt.Visuals2D(critical_curves=critical_curves)

    include_2d = aplt.Include2D(
        mask=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        mapper_data_pixelization_grid=False,
    )

    title_fontsize = 24

    colorbar_image_no_zoom = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax_no_zoom, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax_no_zoom, 3), 2),
    )

    colorbar_image_zoom = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, vmax_zoom, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, vmax_zoom, 3), 2),
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
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_no_zoom),
        colorbar=colorbar_image_no_zoom,
        output=aplt.Output(
            filename=f"{mass_tag}_data",
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
    fit_imaging_plotter.figures_2d(image=True)

    ### Data (Zoomed) ###

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
        title=aplt.Title(label=f"Data ({waveband.upper()})", fontsize=title_fontsize),
        axis=aplt.Axis(extent=[-1.0, 0.5, -1.2, 0.3]),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_zoom),
        colorbar=colorbar_image_zoom,
        output=aplt.Output(
            filename=f"{mass_tag}_data_zoomed",
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
    fit_imaging_plotter.figures_2d(image=True)

    ### Lensed Source Reconstruction ###

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Lensed Source {mass_label} ({waveband.upper()})",
            fontsize=title_fontsize,
        ),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_no_zoom),
        colorbar=colorbar_image_no_zoom,
        output=aplt.Output(
            filename=f"{mass_tag}_image",
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
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, model_image=True)

    ### Lensed Source Reconstruction (Zoomed) ###

    mat_plot_2d = mat_plot_2d_image_zoom + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Lensed Source {mass_label} ({waveband.upper()})",
            fontsize=title_fontsize,
        ),
        axis=aplt.Axis(extent=[-1.0, 0.5, -1.2, 0.3]),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_zoom),
        colorbar=colorbar_image_zoom,
        output=aplt.Output(
            filename=f"{mass_tag}_image_zoomed",
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
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, model_image=True)

    ### Normalized Residual Map ###

    colorbar_norm = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(vmin, vmax, 3), 2),
        manual_tick_labels=np.round(np.linspace(vmin, vmax, 3), 2),
    )

    mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Residuals {mass_label} ({waveband.upper()})",
            fontsize=title_fontsize,
        ),
        cmap=aplt.Cmap(vmin=vmin, vmax=vmax),
        colorbar=colorbar_norm,
        output=aplt.Output(
            filename=f"{mass_tag}_norm",
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

    mat_plot_2d = mat_plot_2d_image_zoom + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Residuals {mass_label} ({waveband.upper()})",
            fontsize=title_fontsize,
        ),
        axis=aplt.Axis(extent=[-1.0, 0.5, -1.2, 0.3]),
        cmap=aplt.Cmap(vmin=vmin, vmax=vmax),
        colorbar=colorbar_norm,
        output=aplt.Output(
            filename=f"{mass_tag}_norm_zoomed",
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

    ### Source Recon ###

    mat_plot_2d_source_zoom = mat_plot_2d_base + aplt.MatPlot2D(
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[0.5, 0.9, 1.3],
            rotation="vertical",
            va="center",
        ),
        xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[0.0, 0.4, 0.8]),
    )

    colorbar_source = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(0.0, 0.1, 3), 2),
        manual_tick_labels=np.round(np.linspace(0.0, 0.1, 3), 2),
    )

    include_2d = aplt.Include2D(
        critical_curves=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        mapper_data_pixelization_grid=False,
        mapper_source_pixelization_grid=False,
        mapper_source_grid_slim=False,
    )

    mat_plot_2d = mat_plot_2d_source_zoom + aplt.MatPlot2D(
        title=aplt.Title(
            label=f"Source {mass_label} ({waveband.upper()})", fontsize=title_fontsize
        ),
        axis=aplt.Axis(extent=[-0.1, 0.9, 0.4, 1.4]),
        cmap=aplt.Cmap(vmin=-0.0, vmax=0.1),
        colorbar=colorbar_source,
        voronoi_drawer=aplt.VoronoiDrawer(edgecolor=None),
        output=aplt.Output(
            filename=f"{mass_tag}_source_recon",
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging, mat_plot_2d=mat_plot_2d, include_2d=include_2d
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)


"""
Plot fits.
"""

search_name = "mass_total[1]_mass[total]_source"

mass_profile_plotter_pl = plot_fit(
    waveband=waveband,
    search_name=search_name,
    mass_cls=al.mp.EllPowerLaw,
    mass_label="PL",
    mass_tag="pl",
    plot_both_crits=True,
)

mass_profile_plotter_bpl = plot_fit(
    waveband=waveband,
    search_name=search_name,
    mass_cls=al.mp.EllPowerLawBroken,
    mass_label="BPL",
    mass_tag="bpl",
    plot_both_crits=False,
)
mass_profile_plotter_pl_smbh = plot_fit(
    waveband=waveband,
    search_name=search_name,
    mass_cls=al.mp.EllPowerLaw,
    smbh_cls=al.mp.PointMass,
    mass_label="PL + SMBH",
    mass_tag="pl_smbh",
    plot_both_crits=False,
)
mass_profile_plotter_bpl_smbh = plot_fit(
    waveband=waveband,
    search_name=search_name,
    mass_cls=al.mp.EllPowerLawBroken,
    smbh_cls=al.mp.PointMass,
    mass_label="BPL + SMBH",
    mass_tag="bpl_smbh",
    plot_both_crits=False,
)
