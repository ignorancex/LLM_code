import argparse
import os

from matplotlib import pyplot as plt
import pandas as pd

from experiment import io


def hist_plots(
    measures_csv,
    output_filepath,
    bins,
    title,
    xlabel,
    ylabel,
    density=False,
    color=None,
    dpi=400,
    x_range=(0.0, 1.0),
    font_size=None,
    pad_inches=0.1,
):
    """The Histogram plots used for visualizing the measures in experiments 1
    and 2.
    """
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    measures = pd.read_csv(measures_csv, header=None)
    col_size = len(measures.columns)
    measures = pd.DataFrame(measures.values.flatten())

    measures.hist(bins=bins, color=color, range=x_range, density=density)

    if '{bins}' in title:
        title = title.replace('{bins}', str(bins))
    if '{bnn_draws}' in title:
        title = title.replace('{bnn_draws}', str(col_size))

    title = title.replace('\n', '\n')

    if '{bins}' in xlabel:
        xlabel = xlabel.replace('{bins}', str(bins))
    if '{bnn_draws}' in xlabel:
        xlabel = xlabel.replace('{bnn_draws}', str(col_size))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(
        io.create_filepath(output_filepath),
        dpi=dpi,
        pad_inches=pad_inches,
    )
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        'measures_csv',
        help='CSV containing measurements.',
    )

    parser.add_argument(
        'output_filepath',
        default=None,
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '-b',
        '--bins',
        default=1000,
        type=int,
        help='The number of bins to use for the histogram.',
    )

    parser.add_argument(
        '--dpi',
        default=400,
        type=int,
        help='The dpi of the histogram.',
    )

    parser.add_argument(
        '--pad_inches',
        default=0.1,
        type=float,
        help='The padding in inches of the histogram.',
    )

    parser.add_argument(
        '--font_size',
        default=14,
        type=int,
        help='The font size of labels on the histogram.',
    )

    parser.add_argument(
        '-c',
        '--color',
        default=None,
        help='The color of the histogram.',
    )

    parser.add_argument(
        '-t',
        '--title',
        default=None,
        help='The title of the histogram.',
    )

    parser.add_argument(
        '-x',
        '--xlabel',
        default='Normalized Euclidean Distance (bins={bins}, {bnn_draws}draws/pred)',
        help='The label along the x axis.',
    )

    parser.add_argument(
        '-y',
        '--ylabel',
        default='Occurrence Count',
        help='The label along the y axis.',
    )

    parser.add_argument(
        '--density',
        action='store_true',
        help='If given, the hidden layers DO NOT use biases (set to zeros)',
    )

    args = parser.parse_args()

    if args.density and args.ylabel == 'Occurrence Count':
        args.ylabel = 'Occurrence Density'

    return args

if __name__ == '__main__':
    hist_plots(**vars(parse_args()))
