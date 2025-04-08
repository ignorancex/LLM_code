import numpy as np
import os
from os import path
import sys

import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())

workspace_path = os.getcwd()

pixel_scales = 0.04

dataset_name = "rjlens"

# waveband = "f390w"
waveband = "f814w"

if waveband in "f390w":
    vmax = 0.08
else:
    vmax = 0.2

plot_path = path.join(workspace_path, "paper", "images", "data_and_sub", waveband)

dataset_path = path.join("paper", "fits")

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, f"a1201_{waveband}_v3_img_cps.fits"),
    pixel_scales=0.04,
)

colorbar_image = aplt.Colorbar(
    manual_tick_values=np.round(np.linspace(0.0, vmax, 3), 2),
    manual_tick_labels=np.round(np.linspace(0.0, vmax, 3), 2),
)

mat_plot_2d = aplt.MatPlot2D(
    axis=aplt.Axis(extent=[-5.0, 5.0, -5.0, 5.0]),
    cmap=aplt.Cmap(vmin=0.0, vmax=vmax),
    title=aplt.Title(label=f"Abell 1201 {waveband.upper()} Image", fontsize=24),
    yticks=aplt.YTicks(
        fontsize=22,
        suffix='"',
        manual_values=[-4.0, 0.0, 4.0],
        rotation="vertical",
        va="center",
    ),
    xticks=aplt.XTicks(fontsize=22, suffix='"', manual_values=[-4.0, 0.0, 4.0]),
    ylabel=aplt.YLabel(label=""),
    xlabel=aplt.XLabel(label=""),
    colorbar=colorbar_image,
    colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
    positions_scatter=aplt.PositionsScatter(s=100),
    output=aplt.Output(
        filename=f"data_{waveband}",
        path=plot_path,
        format=["png", "pdf"],
        format_folder=True,
        bbox_inches="tight",
    ),
)

visuals_2d = aplt.Visuals2D(positions=al.Grid2DIrregular([(0.95, 3.6)]))

array_plotter = aplt.Array2DPlotter(
    array=image, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
)

array_plotter.figure_2d()
