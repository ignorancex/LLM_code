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

plot_path = path.join(workspace_path, "paper", "images", "light_dark_pdf_x3")


def plot_pdf(
    waveband,
    search_name,
    tag,
    envelope_cls=None,
    smbh_cls=None,
):

    agg_query = agg

    agg_query = agg_query.query(agg_query.search.unique_tag == waveband)
    agg_query = agg_query.query(agg_query.search.name == search_name)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.envelope == envelope_cls)
    agg_query = agg_query.query(agg_query.model.galaxies.lens.smbh == smbh_cls)

    samples = list(agg_query.values("samples"))[0]

    print(samples.model.model_component_and_parameter_names)

    labels = samples.model.parameter_labels_with_superscripts

    print(labels)

    labels = [
        r"\epsilon_{\rm 1}^{\rm ext}",
        r"\epsilon_{\rm 2}^{\rm ext}",
        r"\Gamma^{\rm bulge}",
        r"\Gamma^{\rm disk}",
        r"\Gamma^{\rm envelope}",
        r"y^{\rm dark} \, ['']",
        r"x^{\rm dark} \, ['']",
        r"\epsilon_{\rm 1}^{\rm dark}",
        r"\epsilon_{\rm 2}^{\rm dark}",
        r"\theta_{\rm Ein}^{\rm smbh} \, ['']",
        r"\Psi^{\rm bulge} [e^{\rm -} s^{\rm -1}]",
        r"\Psi^{\rm disk} [e^{\rm -} s^{\rm -1}]",
        r"\Psi^{\rm env} [e^{\rm -} s^{\rm -1}]",
        r"\,",
    ]

    labels = [
        r"M_{\rm 200}^{\rm dark} \, [M_{\rm \odot}]",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
        r"\,",
    ]

    gd_samples = MCSamples(
        samples=np.asarray(samples.parameter_lists),
        loglikes=np.asarray(samples.log_likelihood_list),
        weights=np.asarray(samples.weight_list),
        names=samples.model.model_component_and_parameter_names,
        labels=labels,
        sampler="nested",
    )

    import matplotlib as mpl

    rc_sizes = mpl.rcParams["axes.labelsize"] = 22

    gd_plotter = plots.get_subplot_plotter(
        width_inch=24, subplot_size_ratio=1.4, rc_sizes=rc_sizes
    )

    gd_plotter.settings.fontsize = 24
    gd_plotter.settings.axes_fontsize = 22

    yparam = "galaxies_lens_smbh_einstein_radius"

    xparams = samples.model.model_component_and_parameter_names

    xparams.remove(yparam)

    print(xparams)

    param_limits = {
        yparam: [0.25, 0.6],
        "galaxies_lens_shear_elliptical_comps_0": [-0.2, -0.1],
        "galaxies_lens_shear_elliptical_comps_1": [0.15, 0.25],
        "galaxies_lens_dark_mass_at_200": [0.7e14, 2.2e14],
    }

    gd_plotter.rectangle_plot(
        roots=gd_samples,
        yparams=[yparam],
        xparams=xparams,
        param_limits=param_limits,
        filled=True,
        #    ymarkers={"galaxies_lens_dark_mass_at_200": [1e12, 5e14]},
        #    param_limits={"galaxies_lens_dark_mass_at_200": [1e12, 5e14]}
    )

    output_path = path.join(plot_path)

    plt.savefig(path.join(output_path, f"{tag}_image.png"))
    plt.savefig(path.join(output_path, f"{tag}_image.pdf"))

    plt.close()


"""
Make Mass Profile Plotters.
"""

# search_name = "mass_light_dark[1]_light[fixed]_mass[light_dark]_source__stochastic"
#
# plot_pdf(
#     waveband=waveband,
#     search_name=search_name,
#     envelope_cls=al.mp.EllSersic,
#     tag="const",
# )
#
# search_name = "mass_light_dark[1]_light[fixed]_mass[light_dark]_source_grad__stochastic"
#
# plot_pdf(
#     waveband=waveband,
#     search_name=search_name,
#     envelope_cls=al.mp.EllSersicRadialGradient,
#     tag="grad",
# )
#
# search_name = "mass_light_dark[1]_light[fixed]_mass[light_dark]_source_smbh__stochastic"
#
# plot_pdf(
#     waveband=waveband,
#     search_name=search_name,
#     envelope_cls=al.mp.EllSersic,
#     smbh_cls=al.mp.PointMass,
#     tag="const_smbh",
# )

search_name = (
    "mass_light_dark[1]_light[fixed]_mass[light_dark]_source_grad_smbh__stochastic"
)

plot_pdf(
    waveband=waveband,
    search_name=search_name,
    envelope_cls=al.mp.EllSersicRadialGradient,
    smbh_cls=al.mp.PointMass,
    tag="grad_smbh",
)
