import numpy as np

from histogram import HistogramData

from plotter import enc_data_dir, enc_figure_dir

from encs.plots import density_4d
from encs.plots import density_colormap

from re import search


if __name__ == "__main__":
    # Histogram processing
    new_hist3d = HistogramData(
        file_name=enc_data_dir/
           f"2special_od_test_1k_nus_1-00_1-00.py",
        variable_order=['R_sp', 'theta1', 'theta1p']
    )

    fig, ax = density_4d(new_hist3d, colormap=density_colormap['opendata'])

    fig.savefig(enc_figure_dir /
                'supplemental/od_2special_density.png',
                dpi=100)
