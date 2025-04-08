from astropy import cosmology as cosmo
import numpy as np
from os import path
import os

scale = 1.0

workspace_path = os.getcwd()

plot_path = path.join(workspace_path, "paper", "images", "smbh_demo")

import autolens as al
import autolens.plot as aplt

grid_critical_curves = al.Grid2D.uniform(shape_native=(300, 300), pixel_scales=0.04)

grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.04)

mass = al.mp.EllPowerLaw(
    centre=(0.0, 0.0),
    elliptical_comps=(0.108, -0.089),
    einstein_radius=1.523,
    slope=1.8,
)

shear = al.mp.ExternalShear(elliptical_comps=(-0.116, 0.153))

lens = al.Galaxy(redshift=0.5, mass=mass, shear=shear)

source = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllSersic(
        centre=(0.7, 0.22),
        elliptical_comps=(-0.549, -0.272),
        intensity=0.020,
        effective_radius=0.2,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

vmax_image = 0.1

colorbar_image = aplt.Colorbar(
    manual_tick_values=np.round(np.linspace(0.0, vmax_image, 3), 2),
    manual_tick_labels=np.round(np.linspace(0.0, vmax_image, 3), 2),
)

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
    colorbar=colorbar_image,
    colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
)

cmap = aplt.Cmap(vmin=0.0, vmax=vmax_image)
title = aplt.Title(label="Without SMBH", fontsize=title_fontsize)

mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
    cmap=cmap,
    title=title,
    colorbar=colorbar_image,
    output=aplt.Output(
        filename=f"smbh_demo_no_smbh",
        path=plot_path,
        format=["png", "pdf"],
        format_folder=True,
        bbox_inches="tight",
    ),
)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)

tracer_plotter.figures_2d(image=True)

mat_plot_2d_image = aplt.MatPlot2D(
    yticks=aplt.YTicks(
        fontsize=22,
        suffix='"',
        manual_values=[-0.8, 0.0, 0.8],
        rotation="vertical",
        va="center",
    ),
    xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-0.8, 0.0, 0.8]),
    ylabel=aplt.YLabel(label=""),
    xlabel=aplt.XLabel(label=""),
    colorbar=colorbar_image,
    colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
)

mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
    cmap=cmap,
    title=title,
    axis=aplt.Axis(extent=[-scale, scale, -scale, scale]),
    colorbar=colorbar_image,
    output=aplt.Output(
        filename=f"smbh_demo_no_smbh_zoom",
        path=plot_path,
        format=["png", "pdf"],
        format_folder=True,
        bbox_inches="tight",
    ),
)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)

tracer_plotter.figures_2d(image=True)

smbh = al.mp.PointMass(einstein_radius=0.2815)

cosmology = cosmo.Planck15

critical_surface_density = (
    al.util.cosmology.critical_surface_density_between_redshifts_from(
        redshift_0=0.169, redshift_1=0.451, cosmology=cosmo.Planck15
    )
)
einstein_mass = smbh.einstein_mass_angular_from(grid=grid)
einstein_mass_solar_mass = einstein_mass * critical_surface_density

sheet = al.mp.MassSheet(kappa=-0.015)

lens = al.Galaxy(redshift=0.5, mass=mass, shear=shear, smbh=smbh, sheet=sheet)

tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

critical_curves = [tracer.tangential_critical_curve_from(grid=grid_critical_curves)]

visuals_2d = aplt.Visuals2D(critical_curves=critical_curves)

title = aplt.Title(
    label=r"With $M_{\rm BH} = 10^{10}$M$_{\rm \odot}$ SMBH",
    fontsize=title_fontsize,
)

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
    colorbar=colorbar_image,
    colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
)

mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
    cmap=cmap,
    title=title,
    colorbar=colorbar_image,
    output=aplt.Output(
        filename=f"smbh_demo_smbh",
        path=plot_path,
        format=["png", "pdf"],
        format_folder=True,
        bbox_inches="tight",
    ),
)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
)

tracer_plotter.figures_2d(image=True)

mat_plot_2d_image = aplt.MatPlot2D(
    yticks=aplt.YTicks(
        fontsize=22,
        suffix='"',
        manual_values=[-0.8, 0.0, 0.8],
        rotation="vertical",
        va="center",
    ),
    xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-0.8, 0.0, 0.8]),
    ylabel=aplt.YLabel(label=""),
    xlabel=aplt.XLabel(label=""),
    colorbar=colorbar_image,
    colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
)

mat_plot_2d = mat_plot_2d_image + aplt.MatPlot2D(
    cmap=cmap,
    title=title,
    axis=aplt.Axis(extent=[-scale, scale, -scale, scale]),
    colorbar=colorbar_image,
    output=aplt.Output(
        filename=f"smbh_demo_smbh_zoom",
        path=plot_path,
        format=["png", "pdf"],
        format_folder=True,
        bbox_inches="tight",
    ),
)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
)

tracer_plotter.figures_2d(image=True)
