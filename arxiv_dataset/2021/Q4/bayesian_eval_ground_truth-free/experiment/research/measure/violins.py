import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import seaborn as sns

from experiment import io

from psych_metric.metrics import measure


def split_violins(
    train_csv,
    test_csv,
    title=None,
    model_label='Conditional Probability Model IDs',
    measure_label='Normalized Euclidean Distance',
    conditional_models=['conditional_model'],
    #density=True,
    cred_intervals=None,
    measure_range=(0.0, 1.0),
    font_size=None,
    scale='area',
    scale_hue=True,
    bw='scott',
    cut=0.0,
    orient='h',
    alpha=1.0,
    linewidth=None,
    cred_interval_linewidth=1.1,
    num_major_ticks=None,
    num_minor_ticks=None,
    sns_style='whitegrid',
    inner='box',
    legend=True,
    ax=None,
    output_path=None,
    dpi=400,
    pad_inches=0.1,
    mark_lines=None,
    no_inf=True,
    overwrite=False,
    cut_to_measure_range=False,
):
    """The violin plots used for visualizing the measures in experiments 1
    and 2 using seaborn violinplot.

    Parameters
    ----------
    """
    if isinstance(sns_style, str):
        sns.set_style(sns_style)

    # Load all of the csvs.
    if isinstance(train_csv, str) and isinstance(test_csv, str):
        # TODO adjust this to allow plotting for single (train,test) pair.
        train = pd.read_csv(train_csv, header=None).values.flatten()
        test = pd.read_csv(test_csv, header=None).values.flatten()

        if no_inf:
            # isfinite() as idx is fine to do here because of the flatten.
            train_finite = np.isfinite(train)
            if train_finite.sum() == 0:
                raise ValueError('All of train is not finite!')
            train = train[train_finite]

            test_finite = np.isfinite(test)
            if test_finite.sum() == 0:
                raise ValueError('All of test is not finite!')
            test = test[test_finite]
        if cut_to_measure_range:
            train = train[np.logical_and(
                train >= measure_range[0],
                train <= measure_range[1],
            )]
            test = test[np.logical_and(
                test >= measure_range[0],
                test <= measure_range[1],
            )]

        # Create DataFrame for plotting
        measures = pd.DataFrame(
            np.concatenate([train, test]),
            columns=[measure_label],
        )
        measures['data_split'] = ['train'] * len(train) + ['test'] * len(test)
        measures['model'] = conditional_models[0]
    elif (
        isinstance(train_csv, list)
        and isinstance(test_csv, list)
        and len(train_csv) == len(test_csv)
        and conditional_models is not None
        and len(train_csv) == len(conditional_models)
    ):
        measures = []
        model_ids = []
        data_splits = []
        for i in range(len(train_csv)):
            train = pd.read_csv(train_csv[i], header=None).values.flatten()
            test = pd.read_csv(test_csv[i], header=None).values.flatten()

            if no_inf:
                # isfinite() as idx is fine to do here because of the flatten.
                train_finite = np.isfinite(train)
                if train_finite.sum() == 0:
                    raise ValueError('All of train is not finite!')
                train = train[train_finite]

                test_finite = np.isfinite(test)
                if test_finite.sum() == 0:
                    raise ValueError('All of test is not finite!')
                test = test[test_finite]
            if cut_to_measure_range:
                train = train[np.logical_and(
                    train >= measure_range[0],
                    train <= measure_range[1],
                )]
                test = test[np.logical_and(
                    test >= measure_range[0],
                    test <= measure_range[1],
                )]

            measures += [train, test]

            model_ids += [conditional_models[i]] * (len(train) + len(test))
            data_splits += ['train'] * len(train) + ['test'] * len(test)

        # Create DataFrame for plotting
        measures = pd.DataFrame(
            np.concatenate(measures),
            columns=[measure_label],
        )
        measures[model_label] = model_ids
        measures['data_split'] = data_splits
    else:
        raise TypeError(' '.join([
            'Expected either pair of strs, or pair of lists.',
            f'Instead recieved train_csv: {train_csv}',
            f'Instead recieved test_csv: {test_csv}',
        ]))

    # Plot violin
    if orient == 'v':
        ax = sns.violinplot(
            model_label,
            measure_label,
            hue='data_split',
            split=True,
            palette={'train': '#6c8ebf', 'test':'#d6b656'},
            data=measures,
            scale=scale,
            scale_hue=scale_hue,
            bw=bw,
            cut=cut,
            order=conditional_models,
            orient=orient,
            linewidth=linewidth,
            #density=density,
            ax=ax,
            alpha=alpha,
            inner=inner,
        )

        if measure_range is not None:
            ax.set(ylim=measure_range)
        if num_major_ticks is not None:
            ax.yaxis.set_major_locator(LinearLocator(num_major_ticks))
            ax.grid(True, 'major', 'y', linewidth=1.0)
        if num_minor_ticks is not None:
            ax.yaxis.set_minor_locator(LinearLocator(num_minor_ticks))
            ax.grid(True, 'minor', 'y', linewidth=0.5, linestyle='--')

        if isinstance(cred_intervals, list):
            # Plot the given values over the entire violin plot
            # NOTE requires post processing as svg if multiple violins
            # may be able to find ymin and ymax from plot...
            color_switch = False
            for interval in cred_intervals:
                ax.axhline(
                    interval,
                    linewidth=cred_interval_linewidth,
                    color='#d6b656' if color_switch else '#6c8ebf',
                    linestyle='-.',
                )

                if len(cred_intervals) >= 2 * len(conditional_models):
                    color_switch ^= True

        if isinstance(mark_lines, list):
            for mark in mark_lines:
                ax.axhline(
                    mark,
                    linewidth=cred_interval_linewidth,
                    color='red',
                    #linestyle='-.',
                )

    else:
        ax = sns.violinplot(
            measure_label,
            model_label,
            hue='data_split',
            split=True,
            palette={'train': '#6c8ebf', 'test':'#d6b656'},
            data=measures,
            scale=scale,
            scale_hue=scale_hue,
            bw=bw,
            cut=cut,
            order=conditional_models,
            orient=orient,
            linewidth=linewidth,
            #density=density,
            ax=ax,
            alpha=alpha,
            inner=inner,
        )

        if measure_range is not None:
            ax.set(xlim=measure_range)
        if num_major_ticks is not None:
            ax.xaxis.set_major_locator(LinearLocator(num_major_ticks))
            ax.grid(True, 'major', 'x', linewidth=1.0)
        if num_minor_ticks is not None:
            ax.xaxis.set_minor_locator(LinearLocator(num_minor_ticks))
            ax.grid(True, 'minor', 'x', linewidth=0.5, linestyle='--')

        if isinstance(cred_intervals, list):
            # Plot the given values over the entire violin plot
            # NOTE requires post processing as svg if multiple violins
            # may be able to find ymin and ymax from plot...
            color_switch = False
            for interval in cred_intervals:
                ax.axvline(
                    interval,
                    linewidth=cred_interval_linewidth,
                    color='#d6b656' if color_switch else '#6c8ebf',
                    linestyle='-.',
                )

                if len(cred_intervals) >= 2 * len(conditional_models):
                    color_switch ^= True

        if isinstance(mark_lines, list):
            for mark in mark_lines:
                ax.axvline(
                    mark,
                    linewidth=cred_interval_linewidth,
                    color='red',
                    #linestyle='-.',
                )

    if title is not None:
        #plt.title(title)
        ax.set_title(title)
    if not legend:
        ax.get_legend().set_visible(False)


    if isinstance(output_path, str):
        plt.savefig(
            io.create_filepath(output_path, overwrite=overwrite),
            dpi=dpi,
            pad_inches=pad_inches,
        )
        plt.close()

    return ax


def get_cred_intervals(filepath, keys, concat_intervals=None):
    """Gets cred intervals in order of keys given from JSON."""
    with open(filepath, 'r') as f:
        content = json.load(f)
        cred_intervals = []

        # loop through all keys (conditional predictors)
        for key in keys:
            if key not in content:
                continue

            # loop through train and test and possibly val sets
            if (
                isinstance(content[key]['train'], dict)
                and isinstance(content[key]['train'], dict)
            ):
                cred_intervals.append(content[key]['train']['lower_quantile'])
                cred_intervals.append(content[key]['test']['lower_quantile'])
                cred_intervals.append(content[key]['train']['higher_quantile'])
                cred_intervals.append(content[key]['test']['higher_quantile'])
            else:
                cred_intervals.append(content[key]['train'])
                cred_intervals.append(content[key]['test'])

        if concat_intervals is not None:
            cred_intervals += concat_intervals

    return cred_intervals


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        'train_paths',
        default=None,
        nargs='+',
        type=str,
        help='CSV containing measurements on train.',
    )

    parser.add_argument(
        '--test_paths',
        default=None,
        nargs='+',
        type=str,
        help='CSV containing measurements on test.',
    )

    parser.add_argument(
        '--conditional_models',
        default=None,
        nargs='+',
        type=str,
        help='Names of the different conditional models.',
    )

    parser.add_argument(
        '--output_path',
        default=None,
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If given, the script will overwrite pre-existing files.',
    )

    #parser.add_argument(
    #    '-b',
    #    '--bins',
    #    default=1000,
    #    type=int,
    #    help='The number of bins to use for the histogram.',
    #)

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

    #parser.add_argument(
    #    '-c',
    #    '--color',
    #    default=None,
    #    help='The color of the histogram.',
    #)

    parser.add_argument(
        '--orient',
        default='h',
        help='The orientation of the violin plot.',
    )

    parser.add_argument(
        '-t',
        '--title',
        default=None,
        help='The title of the violin plot.',
    )

    parser.add_argument(
        '--measure_label',
        default='Normalized Euclidean Distance',
        help='The label along the x axis.',
    )

    parser.add_argument(
        '--model_label',
        default='Conditional Probability Model IDs',
        help='The label along the y axis.',
    )

    parser.add_argument(
        '--not_density',
        action='store_true',
        help='If given, the hidden layers DO NOT use biases (set to zeros)',
    )

    parser.add_argument(
        '--legend',
        action='store_true',
        help='If given, the script will include a legend on the plot.',
    )

    parser.add_argument(
        '--measure_lower',
        default=None,
        type=float,
        help='The lower bound of the measure.',
    )

    parser.add_argument(
        '--measure_upper',
        default=None,
        type=float,
        help='The upper bound of the measure.',
    )

    parser.add_argument(
        '--cut_to_measure_range',
        action='store_true',
        help='If given, all elements outside of the range are removed.',
    )

    parser.add_argument(
        '--num_major_ticks',
        default=11,
        type=int,
        help='The number of major ticks in the violin plot.',
    )

    parser.add_argument(
        '--num_minor_ticks',
        default=21,
        type=int,
        help='The number of minor ticks in the violin plot.',
    )

    parser.add_argument(
        '--linewidth',
        default=0.1,
        type=float,
        help='The line width of the violin plot.',
    )

    parser.add_argument(
        '--cred_interval_linewidth',
        default=1.1,
        type=float,
        help='The line width of the violin plot.',
    )

    parser.add_argument(
        '--cred_intervals',
        default=None,
        nargs='+',
        type=float,
        help='The credibility interval to overlay the violin plot.',
    )

    parser.add_argument(
        '--cred_intervals_json',
        default=None,
        help='JSON containing credibility intervals to put on violin plot.',
    )

    parser.add_argument(
        '--mark_lines',
        default=None,
        nargs='+',
        type=float,
        help='Lines of importance to be marked on plot on measure axis',
    )

    parser.add_argument(
        '--inner',
        default='box',
        choices=['box', 'quartile', 'point', 'stick', 'None'],
        help='The line width of the violin plot.',
    )

    parser.add_argument(
        '--scale',
        default='area',
        choices=['area', 'width', 'count'],
        help='The line width of the violin plot.',
    )

    parser.add_argument(
        '--no_scale_hue',
        action='store_true',
        help='If given, seaborn violinplot scale_hue set to False.',
    )

    parser.add_argument(
        '--cut',
        default=0.0,
        type=float,
        help='The seaborn violin plot cut param. Defaults to stop at extrema.',
    )

    parser.add_argument(
        '--bw',
        default='scott',
        type=str,
        help='The seaborn violin plot bandwidth param.',
    )

    args = parser.parse_args()

    # Handle args post parsing
    args.density = not args.not_density
    del args.not_density

    #if args.density and args.ylabel == 'Occurrence Count':
    #    args.ylabel = 'Occurrence Density'

    # if bins: Measure label: (bins={bins}, {bnn_draws}draws/pred)

    if args.measure_lower is None and args.measure_upper is None:
        args.measure_range = None
        #TODO pass measure_range
    else:
        args.measure_range = (args.measure_lower, args.measure_upper)

    if args.inner is 'None':
        args.inner = None

    args.scale_hue = not args.no_scale_hue
    del args.no_scale_hue

    if args.bw != 'scott' and args.bw != 'silverman':
        args.bw = float(args.bw)

    #TODO if args.test_paths is None: del the test portion of the violin

    # If given cred_intervals_json put at beginning of cred_intevals
    if args.cred_intervals_json is not None:
        args.cred_intervals = get_cred_intervals(
            args.cred_intervals_json,
            args.conditional_models,
            args.cred_intervals,
        )

    return args


if __name__ == '__main__':
    args = parse_args()

    sns.set_style('whitegrid')

    fig, ax = plt.subplots(1, 1, figsize=(4,4))

    ax = split_violins(
        args.train_paths,
        args.test_paths,
        title=args.title,
        model_label=args.model_label,
        conditional_models=args.conditional_models,
        orient=args.orient,
        linewidth=args.linewidth,
        cred_intervals=args.cred_intervals,
        cred_interval_linewidth=args.cred_interval_linewidth,
        measure_range=args.measure_range,
        scale=args.scale,
        scale_hue=args.scale_hue,
        bw=args.bw,
        cut=args.cut,
        overwrite=args.overwrite,
        num_major_ticks=args.num_major_ticks,
        num_minor_ticks=args.num_minor_ticks,
        ax=ax,
        legend=args.legend,
        measure_label=args.measure_label,
        inner=args.inner,
        mark_lines=args.mark_lines,
        cut_to_measure_range=args.cut_to_measure_range,
    )

    output_path = io.create_filepath(args.output_path, overwrite=args.overwrite)
    fig.savefig(
        output_path,
        dpi=args.dpi,
        pad_inches=args.pad_inches,
        bbox_inches='tight',
    )
