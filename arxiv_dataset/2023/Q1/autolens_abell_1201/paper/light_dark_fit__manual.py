import numpy as np
import os
from os import path
from matplotlib.patches import Circle
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

waveband = "f390w"
# waveband = "f814w"

if waveband in "f390w":
    vmax_zoom = 0.02
else:
    vmax_zoom = 0.2

if waveband in "f390w":
    vmax_no_zoom = 0.07
else:
    vmax_no_zoom = 0.15

# plot_path = path.join(workspace_path, "paper", "images", "light_dark_fit_x3_radial_decomp", waveband)

plot_path = path.join("paper", "images", "decomposed_radial_manual")


"""
__Grid__
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.04)

vmax = 5.0
vmin = -5.0


def plot_fit(
    waveband,
    search_name,
    mass_label,
    mass_tag,
    bulge_cls,
    disk_cls=None,
    envelope_cls=None,
    smbh_cls=None,
    plot_crits=True
):

    agg_query = agg

    agg_query = agg_query.query(agg_query.search.unique_tag == waveband)
    agg_query = agg_query.query(agg_query.search.name == search_name)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.bulge == bulge_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.disk == disk_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.envelope == envelope_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.smbh == smbh_cls)

    samples = list(agg_query.values("samples"))[0]

    instance = samples.max_log_likelihood_instance

    bulge = al.lmp.EllSersicRadialGradient(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0309506766816977, -0.0621964387481318),
        intensity=0.22305566711518723,
        effective_radius=0.4624170312787752,
        sersic_index=1.2671926982808182,
        mass_to_light_ratio=8.0,
        mass_to_light_gradient=-0.5
    )

    disk = al.lmp.EllSersicRadialGradient(
        centre=(0.0, 0.0),
        elliptical_comps=(0.16208666776894873, -0.14256415128388042),
        intensity=0.02901283096677522,
        effective_radius=5.131096088458944,
        sersic_index=1.3062257408698343,
        mass_to_light_ratio=6.5,
        mass_to_light_gradient=-0.5
    )

    dark = al.mp.EllNFWMCRLudlow(
        centre=(0.0, 0.0),
        elliptical_comps=(0.007, 0.094),
        mass_at_200=41998984274016.766,
        redshift_object=0.169,
        redshift_source=0.451,
    )

    hyper_model_image = list(agg_query.values("hyper_model_image"))[0]
    hyper_galaxy_image_path_dict = list(agg_query.values("hyper_galaxy_image_path_dict"))[0]

    instance.galaxies.source.hyper_model_image = hyper_model_image
    instance.galaxies.source.hyper_galaxy_image = hyper_model_image

    tracer = al.Tracer.from_galaxies(galaxies=instance.galaxies)

    tracer.galaxies[0].bulge = bulge
    tracer.galaxies[0].disk = disk
    tracer.galaxies[0].dark = dark

    sparse_grids_of_planes = list(agg_query.values("preload_sparse_grids_of_planes"))[0]

    preloads = al.Preloads(
        sparse_image_plane_grid_pg_list=sparse_grids_of_planes,
        use_w_tilde=False,
    )

    if len(preloads.sparse_image_plane_grid_pg_list) == 2:
        if type(preloads.sparse_image_plane_grid_pg_list[1]) != list:
            preloads.sparse_image_plane_grid_pg_list[1] = [
                preloads.sparse_image_plane_grid_pg_list[1]
            ]

    imaging = al.agg.ImagingAgg(aggregator=agg_query)
    imaging = list(imaging.imaging_gen_from())[0]

    fit_imaging = al.FitImaging(dataset=imaging, tracer=tracer, preloads=preloads)

    include_2d = aplt.Include2D(
        mask=False,
        critical_curves=plot_crits,
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

    mat_plot_2d = mat_plot_2d_image +  aplt.MatPlot2D(
     #   axis=aplt.Axis(extent=[-3.7, 3.7, -3.7, 3.7]),
        title=aplt.Title(label=f"Data ({waveband.upper()})", fontsize=title_fontsize),
        ylabel=aplt.YLabel(labelpad=0.5),
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='"',
            manual_values=[-3.0, 0.0, 3.0],
            rotation="vertical",
            va="center",
        ),
        colorbar=colorbar_image_no_zoom,
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_no_zoom),
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


    mat_plot_2d =mat_plot_2d_image_zoom + aplt.MatPlot2D(
        title=aplt.Title(label=f"Data ({waveband.upper()})", fontsize=title_fontsize),
        ylabel=aplt.YLabel(labelpad=0.5),
        colorbar=colorbar_image_zoom,
        axis=aplt.Axis(extent=[-1.0, 0.5, -1.2, 0.3]),
        cmap=aplt.Cmap(vmin=0.0, vmax=vmax_zoom),
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
    )
    fit_imaging_plotter.figures_2d(image=True)

    ### Lensed Source Reconstruction ###

    mat_plot_2d = mat_plot_2d_image +  aplt.MatPlot2D(
        title=aplt.Title(label=f"Lensed Source {mass_label} ({waveband.upper()})", fontsize=title_fontsize),
        ylabel=aplt.YLabel(labelpad=0.5),
        cmap=aplt.Cmap(vmin=0.0),
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
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, model_image=True)

    ### Lensed Source Reconstruction (Zoomed) ###

    mat_plot_2d = mat_plot_2d_image_zoom +aplt.MatPlot2D(
        title=aplt.Title(label=f"Lensed Source {mass_label} ({waveband.upper()})", fontsize=title_fontsize),
        ylabel=aplt.YLabel(labelpad=0.5),
        axis=aplt.Axis(extent=[-1.0, 0.5, -1.2, 0.3]),
        cmap=aplt.Cmap(vmin=0.0),
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
    )
    fit_imaging_plotter.figures_2d(normalized_residual_map=True)

    ### Normalized Residual Map (Zoomed) ###

    colorbar_norm = aplt.Colorbar(
        manual_tick_values=np.round(np.linspace(vmin, vmax, 3), 2),
        manual_tick_labels=np.round(np.linspace(vmin, vmax, 3), 2),
    )

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
    )
    fit_imaging_plotter.figures_2d(normalized_residual_map=True)

    ### Source Recon (Zoom) ###

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

    mat_plot_2d = mat_plot_2d_source_zoom +  aplt.MatPlot2D(
        title=aplt.Title(label=f"Source {mass_label} ({waveband.upper()})", fontsize=title_fontsize),
        ylabel=aplt.YLabel(labelpad=0.5),
        axis=aplt.Axis(extent=[-0.1, 0.9, 0.4, 1.4]),
        cmap=aplt.Cmap(vmin=-0.0, vmax=0.1),
        colorbar=colorbar_source,
        voronoi_drawer=aplt.VoronoiDrawer(edgecolor=None),
        output=aplt.Output(
            filename=f"{mass_tag}_source_recon_zoomed",
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

    ## Source Reconstruction ##

    vmax_source = 0.1

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
        fit=fit_imaging, mat_plot_2d=mat_plot_2d, include_2d=include_2d, # visuals_2d=visuals_2d
    )
    fit_imaging_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
Plot fits.
"""

search_name = "mass_light_dark[1]_light[fixed]_mass[light_dark]_source"

plot_fit(
    waveband=waveband,
    search_name=f"{search_name}",
    bulge_cls=al.mp.EllSersic,
    disk_cls=al.mp.EllSersic,
    envelope_cls=None,
    mass_label="Without SMBH",
    mass_tag="grad",
    plot_crits=True,
)

# plot_fit(
#     waveband=waveband,
#     search_name=f"{search_name}_smbh",
#     bulge_cls=al.mp.EllSersic,
#     disk_cls=al.mp.EllSersic,
#     envelope_cls=al.mp.EllSersic,
#     smbh_cls=al.mp.PointMass,
#     mass_label="With SMBH",
#     mass_tag="grad_smbh",
#  #   plot_crits=False
# )

# plot_fit(
#     waveband=waveband,
#     search_name=search_name,
#     bulge_cls=al.mp.EllSersic,
#     disk_cls=al.mp.EllSersic,
#     envelope_cls=al.mp.EllSersic,
#     mass_label="Constant MLR",
#     mass_tag="const",
# )

# plot_fit(
#     waveband=waveband,
#     search_name=f"{search_name}_smbh",
#     bulge_cls=al.mp.EllSersic,
#     disk_cls=al.mp.EllSersic,
#     envelope_cls=al.mp.EllSersic,
#     smbh_cls=al.mp.PointMass,
#     mass_label="Constant MLR + SMBH",
#     mass_tag="const_smbh",
# )