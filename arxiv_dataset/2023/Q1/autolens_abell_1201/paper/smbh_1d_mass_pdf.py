from astropy import cosmology as cosmo
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path

from getdist import MCSamples
from getdist import plots

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
    filename="rjlens_no_lens_light_pdf.sqlite", completed_only=False
)

waveband = "f390w"
# waveband = "f814w"

plot_path = path.join(workspace_path, "paper", "images", "smbh_1d_pdf", waveband)


def sample_ld_from(
    waveband,
    search_name,
    envelope_cls=None,
    smbh_cls=None,
):

    agg_query = agg

    agg_query = agg_query.query(agg_query.search.unique_tag == waveband)
    agg_query = agg_query.query(agg_query.search.name == search_name)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.envelope == envelope_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.smbh == smbh_cls)

    return list(agg_query.values("samples"))[0]


def sample_total_from(waveband, search_name, mass_cls, smbh_cls=None):

    agg_query = agg

    agg_query = agg_query.query(agg_query.search.unique_tag == waveband)
    agg_query = agg_query.query(agg_query.search.name == search_name)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.mass == mass_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.smbh == smbh_cls)

    return list(agg_query.values("samples"))[0]


"""
Make Mass Profile Plotters.
"""

critical_surface_density = (
    al.util.cosmology.critical_surface_density_between_redshifts_from(
        redshift_0=0.169, redshift_1=0.451, cosmology=cosmo.Planck15
    )
)

search_name = (
    "mass_light_dark[1]_light[fixed]_mass[light_dark]_source_grad_smbh__stochastic"
)

samples_ld_x2 = sample_ld_from(
    waveband=waveband,
    search_name=search_name,
    envelope_cls=None,
    smbh_cls=al.mp.PointMass,
)

for sample in samples_ld_x2.sample_list:

    einstein_radius = sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")]

    sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")] = (
        critical_surface_density * np.pi * einstein_radius**2.0
    )

search_name = (
    "mass_light_dark[1]_light[fixed]_mass[light_dark]_source_grad_smbh__stochastic"
)

samples_ld_x3 = sample_ld_from(
    waveband=waveband,
    search_name=search_name,
    envelope_cls=al.mp.EllSersicRadialGradient,
    smbh_cls=al.mp.PointMass,
)

factor = 20.0

for sample in samples_ld_x3.sample_list:
    einstein_radius = sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")]

    sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")] = (
        critical_surface_density * np.pi * einstein_radius**2.0
    )

search_name = "mass_total[1]_mass[total]_source__stochastic"

samples_pl = sample_total_from(
    waveband=waveband,
    search_name=search_name,
    mass_cls=al.mp.EllPowerLaw,
    smbh_cls=al.mp.PointMass,
)

for sample in samples_pl.sample_list:
    einstein_radius = sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")]

    sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")] = (
        critical_surface_density * np.pi * einstein_radius**2.0
    )

samples_bpl = sample_total_from(
    waveband=waveband,
    search_name=search_name,
    mass_cls=al.mp.EllPowerLawBroken,
    smbh_cls=al.mp.PointMass,
)

for sample in samples_bpl.sample_list:
    einstein_radius = sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")]

    sample.kwargs[("galaxies", "lens", "smbh", "einstein_radius")] = (
        critical_surface_density * np.pi * einstein_radius**2.0
    )


gd_samples_list = [
    MCSamples(
        samples=np.asarray(samples.parameter_lists),
        loglikes=np.asarray(samples.log_likelihood_list),
        weights=np.asarray(samples.weight_list),
        names=samples.model.model_component_and_parameter_names,
        labels=[
            label.replace(r"\theta_{\rm Ein}^{\rm smbh}", r"M_{\rm BH}^{\rm smbh}")
            for label in samples.model.parameter_labels_with_superscripts
        ],
        sampler="nested",
    )
    for samples in [samples_ld_x3, samples_ld_x2, samples_pl]
]

print(gd_samples_list)

#   gd_plotter = plots.get_subplot_plotter(width_inch=24, subplot_size_ratio=1.4)

import matplotlib as mpl

rc_sizes = mpl.rcParams["axes.labelsize"] = 40

gd_plotter = plots.get_single_plotter(width_inch=10, rc_sizes=rc_sizes)

gd_plotter.settings.legend_loc = "upper left"
gd_plotter.settings.axis_tick_powerlimits = (-2, 7)
gd_plotter.settings.fontsize = 40
# gd_plotter.settings.axes_fontsize = 24
gd_plotter.settings.legend_frame = False
gd_plotter.settings.prob_label = "P"
gd_plotter.settings.axis_tick_step_groups = [[1, 2, 3, 4, 5]]
#
# {
#     "scaling": True,
#     "scaling_reference_size": 3.5,
#     "scaling_max_axis_size": 3.5,
#     "scaling_factor": 2,
#     "direct_scaling": False,
#     "plot_meanlikes": False,
#     "prob_label": "P",
#     "norm_prob_label": "P",
#     "prob_y_ticks": False,
#     "norm_1d_density": False,
#     "line_styles": [
#         "-k",
#         "-r",
#         "-b",
#         "-g",
#         "-m",
#         "-c",
#         "-y",
#         "--k",
#         "--r",
#         "--b",
#         "--g",
#         "--m",
#     ],
#     "plot_args": None,
#     "line_dash_styles": {"--": (3, 2), "-.": (4, 1, 1, 1)},
#     "line_labels": True,
#     "num_shades": 80,
#     "shade_level_scale": 1.8,
#     "progress": False,
#     "fig_width_inch": 10,
#     "tight_layout": True,
#     "constrained_layout": False,
#     "no_triangle_axis_labels": True,
#     "colormap": "Blues",
#     "colormap_scatter": "jet",
#     "colorbar_tick_rotation": None,
#     "colorbar_label_pad": 0,
#     "colorbar_label_rotation": -90,
#     "colorbar_axes_fontsize": 11,
#     "subplot_size_inch": 10,
#     "subplot_size_ratio": 0.75,
#     "param_names_for_labels": None,
#     "legend_colored_text": False,
#     "legend_loc": "best",
#     "legend_frac_subplot_margin": 0.05,
#     "legend_fontsize": 12,
#     "legend_frame": False,
#     "legend_rect_border": False,
#     "figure_legend_loc": "upper center",
#     "figure_legend_frame": True,
#     "figure_legend_ncol": 0,
#     "linewidth": 1,
#     "linewidth_contour": 0.6,
#     "linewidth_meanlikes": 0.5,
#     "num_plot_contours": 2,
#     "solid_contour_palefactor": 0.6,
#     "solid_colors": [
#         "#006FED",
#         "#E03424",
#         "gray",
#         "#009966",
#         "#000866",
#         "#336600",
#         "#006633",
#         "m",
#         "r",
#     ],
#     "alpha_filled_add": 0.85,
#     "alpha_factor_contour_lines": 0.5,
#     "shade_meanlikes": False,
#     "axes_fontsize": 22,
#     "axes_labelsize": 14,
#     "axis_marker_color": "gray",
#     "axis_marker_ls": "--",
#     "axis_marker_lw": 0.5,
#     "axis_tick_powerlimits": (-2, 7),
#     "axis_tick_max_labels": 7,
#     "axis_tick_step_groups": [[1, 2, 3, 4, 5]],
#     "axis_tick_x_rotation": 0,
#     "axis_tick_y_rotation": 0,
#     "scatter_size": 3,
#     "fontsize": 40,
#     "title_limit": 0,
#     "title_limit_labels": True,
#     "title_limit_fontsize": None,
# }

print(gd_plotter.settings)

gd_plotter.plot_1d(
    roots=gd_samples_list,
    param="galaxies_lens_smbh_einstein_radius",
    lims=[0.1e10, 6e10],
)

gd_plotter.add_legend(
    legend_labels=[
        "Decomposed (x3 Sersic)",
        "Decomposed (x2 Sersic)",
        "Power Law",
        #     "Broken Power Law"
    ],
    fontsize=16,
    frameon=False
    #    re
)

os.makedirs(plot_path, exist_ok=True)

output_path = path.join(plot_path)

plt.savefig(path.join(output_path, f"smbh_1d_mass_pdf.png"))
plt.savefig(path.join(output_path, f"smbh_1d_mass_pdf.pdf"))

plt.close()
