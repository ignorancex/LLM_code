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

agg = af.Aggregator.from_database(filename="rjlens.sqlite", completed_only=True)

plot_path = path.join(workspace_path, "paper", "images", "light_1d")

"""
__Grid__
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.04)

grid.radial_projected_shape_slim = 50


def plot_image(
    waveband,
    search_name,
    filename,
    bulge_cls,
    disk_cls=None,
    envelope_cls=None,
    sersic_index_range=None,
    legend_labels=None,
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

    tracer_agg = al.agg.TracerAgg(aggregator=agg_query)
    tracer_list = list(tracer_agg.randomly_drawn_via_pdf_gen_from(total_samples=100))

    galaxy_list = [tracer.galaxies[0] for tracer in tracer_list[0]]

    include_1d = aplt.Include1D(einstein_radius=False, half_light_radius=False)

    title_fontsize = 24

    mat_plot_1d = aplt.MatPlot1D(
        title=aplt.Title(
            label=f"Intensity vs Radius ({waveband.upper()})", fontsize=title_fontsize
        ),
#        ylabel=aplt.YLabel(label=r"Intensity ($e^{-}\,s^{-1}$)", labelpad=1.0, fontsize=22),
        ylabel=aplt.YLabel(label=""),
        axis=aplt.Axis(extent=[None, None, 1e-3, 10.0]),
        yx_plot=aplt.YXPlot(plot_axis_type="semilogy"),
        xlabel=aplt.XLabel(label=""),
        yticks=aplt.YTicks(
            fontsize=22,
            suffix='',
            manual_values=[1e-3, 1e-2, 1e-1, 1e0, 1e1],
            rotation="vertical",
            va="center",
        ),
        legend=aplt.Legend(fontsize=22, frameon=False),
        xticks=aplt.XTicks(fontsize=22, suffix='"',
                           manual_values=[0.0, 0.5, 1.0, 1.5, 2.0],
                      #    manual_units=[0.0, 0.5, 1.0, 1.5, 2.0]
                           ),
        output=aplt.Output(
            filename=filename,
            path=plot_path,
            format=["png", "pdf"],
            format_folder=True,
            bbox_inches="tight",
        ),
    )

    galaxy_plotter = aplt.GalaxyPDFPlotter(
        galaxy_pdf_list=galaxy_list,
        grid=grid,
        mat_plot_1d=mat_plot_1d,
        include_1d=include_1d,
    )

    galaxy_plotter.figures_1d_decomposed(image=True, legend_labels=legend_labels)


waveband = "f814w"
search_name = "light[1]_light[parametric]"

plot_image(
    waveband=waveband,
    search_name=search_name,
    bulge_cls=al.lp.EllSersic,
    filename="light_1d_sersic_x1",
    legend_labels=["Total", "Bulge"],
)

plot_image(
    waveband=waveband,
    search_name=search_name,
    bulge_cls=al.lp.EllSersic,
    disk_cls=al.lp.EllSersic,
    filename="light_1d_align_all_x2",
    sersic_index_range=[0.995, 0.997],
    legend_labels=["Total", "Bulge", "Disk"],
)

plot_image(
    waveband=waveband,
    search_name=search_name,
    bulge_cls=al.lp.EllSersic,
    disk_cls=al.lp.EllSersic,
    filename="light_1d_align_centre_x2",
    sersic_index_range=[1.233, 1.235],
    legend_labels=["Total", "Bulge", "Disk"],
)

plot_image(
    waveband=waveband,
    search_name=search_name,
    bulge_cls=al.lp.EllSersic,
    disk_cls=al.lp.EllSersic,
    filename="light_1d_no_align_x2",
    sersic_index_range=[1.266, 1.268],
    legend_labels=["Total", "Bulge", "Disk"],
)

# plot_image(
#     waveband=waveband,
#     search_name=search_name,
#     bulge_cls=al.lp.EllSersic,
#     disk_cls=al.lp.EllSersic,
#     envelope_cls=al.lp.EllSersic,
#     filename="light_1d_no_align_x3",
#     legend_labels=["Total", "Bulge", "Disk", "Envelope"],
# )
