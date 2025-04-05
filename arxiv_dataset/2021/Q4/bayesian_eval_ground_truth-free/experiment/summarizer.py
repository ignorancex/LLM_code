"""
script for refactoring the results into the expected format with `annotation
aggregation.csv`, and the expected directory structure where it follows:
`data_collection/dataset_id/method/method_params/seed/outputs`
"""

# NOTE this is unnecessary if I rerun it all with updated output saving.

import csv
from datetime import datetime
from math import sqrt
import os
import json

import numpy as np
from scipy.stats import multinomial, entropy, wasserstein_distance
from sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt

from psych_metric.datasets import data_handler

def model_parameter_summarize(
    input_dir,
    summary_csv='summary.csv',
    annotation_aggregation_csv='annoatation_aggregation.csv',
    metrics_file='metrics.json',
    output='summary.json',
    quantiles=[0, 0.25, 0.5, 0.75, 1],
):
    """Traverses the given root directory to find directories that contain a
    summary.csv. then combines the summaries together forming the root directory
    summary.

    Parameters
    ----------
    input_dir : str
        The filepath to the input directory to be traversed to find directories
        that contain the results of annotation aggregation methods.
    summary : str, optional
        The filename of the summary csv
    """
    dir_queue = [input_dir]
    summaries = []

    # Traverse through the directory tree with input_dir as root dir
    while dir_queue:
        focus_dir = dir_queue.pop()
        dir_files = [os.path.join(focus_dir, dir_item) for dir_item in os.listdir(focus_dir)]

        # needs both the corresponding summary.csv and annotation_aggregation.csv
        summary_file = None
        annotation_aggregation_file = None
        metrics_file = None # NOTE assumes metrics will exist, ignores otherwise.

        # Check for 3 files; add dirs to queue
        while dir_files:
            focus_file = dir_files.pop()

            if os.is_file(focus_file):
                focus_file_end = focus_file.rsplit(os.path.sep)[-1]

                if summary_csv == focus_file_end:
                    summary_file = focus_file

                elif annotation_aggregation_csv == focus_file_end:
                    annotation_aggregation_file = focus_file

                elif metrics_json == focus_file_end:
                    metrics_file = focus_file

            elif os.is_dir(focus_file):
                dir_queue = [focus_file] + dir_queue

        if summary_file is not None and annotation_aggregation_file is not None and metrics_file is not None:
            summaries.append([summary_file, metrics_file])

    # Summaries have been gathered. Now summarize the summaries.
    top_level_summary = dict()

    # Set intial values
    top_level_summary['runtime_process'] = []
    top_level_summary['runtime_performance'] = []
    top_level_summary['earliest_datetime'] = datetime.max
    top_level_summary['latest_datetime'] = datetime.min

    # if metrics exist, combine into summary
    top_level_summary['metrics'] = dict()

    first = True # temporary, assumes that all summaries are for the same method instance on same datset, etc.
    for summary_file, metrics_file in summaries:
        read_summary = read_csv_to_dict(None, summary_file)

        if first: # TODO Assumes that the summaries are for the same thing, need to perform a check
            top_level_summary['truth_inference_method'] = read_summary['truth_inference_method']
            top_level_summary['model_parameters'] = read_summary['model_parameters']
            top_level_summary['dataset'] = read_summary['dataset']
            top_level_summary['dataset_filepath'] = read_summary['dataset_filepath']
            #top_level_summary['task_type'] =
            first = False

        # Save the runtime distribution.
        top_level_summary['runtime_process'].append(read_summary['runtime_process'])
        top_level_summary['runtime_performance'].append(read_summary['runtime_process'])

        # Save min and max datetime of summarized summaries.
        date_time = datetime.strptime(read_summary['datetime'], '%Y-%m-%d %H:%M:%S.%f')
        top_level_summary['earliest_datetime'] = min(top_level_summary['earliest_datetime'], date_time)
        top_level_summary['latest_datetime'] = max(top_level_summary['latest_datetime'], date_time)

        # read the metrics file
        with open(metrics_file, 'r') as read_metrics:
            metrics = json.load(read_metrics)
            metrics.pop('meta')

            # Add results in single list for all metrics
            for metric, result in metrics.items():
                if metric not in top_level_summary['metrics']:
                    top_level_summary['metrics'][metric] = [result]
                else:
                    top_level_summary['metrics'][metric].append(result)

    # Calculate stats for runtime process
    # mean, sd
    top_level_summary['runtime_process']['mean'] = np.mean(top_level_summary['runtime_process'])
    top_level_summary['runtime_process'][metric]['sd'] = np.std(top_level_summary['runtime_process'])
    # median, quantiles, min, max
    metric_quantiles = np.quantiles(top_level_summary['runtime_process'], quantiles)
    top_level_summary['runtime_process']['quantiles'] = {q:metric_quantiles[i] for i, q in quantiles}

    # Calculate stats for runtime performance
    top_level_summary['runtime_performance']['mean'] = np.mean(top_level_summary['runtime_performance'])
    top_level_summary['runtime_performance'][metric]['sd'] = np.std(top_level_summary['runtime_performance'])
    # median, quantiles, min, max
    metric_quantiles = np.quantiles(top_level_summary['runtime_performance'], quantiles)
    top_level_summary['runtime_performance']['quantiles'] = {q:metric_quantiles[i] for i, q in quantiles}

    # Convert datetime objects into str equivalents keeping with summary standard.
    top_level_summary['earliest_datetime'] = str(top_level_summary['earliest_datetime'])
    top_level_summary['latest_datetime'] = str(top_level_summary['latest_datetime'])

    # loop through metrics and get each metric's statistics:
    for metric, results in top_level_summary['metrics'].items():
        # mean, sd
        top_level_summary['metrics'][metric]['mean'] = np.mean(results)
        top_level_summary['metrics'][metric]['sd'] = np.std(results)
        # median, quantiles, min, max
        metric_quantiles = np.quantiles(results, quantiles)
        top_level_summary['metrics'][metric]['quantiles'] = {q:metric_quantiles[i] for i, q in quantiles}
        # mode
        #mode = list(pd.Series(results).mode())
        #top_level_summary['metrics'][metric]['mode'] = None if len(mode) == len(results) else mode

    # Save the top level summary as json
    with open(output, 'w') as results_file:
        json.dump(metric_results, results_file, indent=4)

def parse_args():
    """Parses the arguments when this script is called.

    Returns
    -------
    argparse.Namespace
        The configuration variables for the experiments to be run.
    """
    parser = argparse.ArgumentParser(description=' '.join([
        'Traverse the directory tree of the given directory and calculates',
        'the desired metrics of the saved annotation aggregation csv when',
        'compared to the ground truth of the dataset.',
    ]})

    parser.add_argument(
        'input_dir',
        default=None,
        help='Enter the file path to the root input directory.',
    )

    parser.add_argument(
        '-s',
        '--summary_csv',
        default='summary.csv',
        help='The expected filename of the summary csv.',
    )

    parser.add_argument(
        '-a',
        '--annotation_aggregation_csv',
        default='annotation_aggregation.csv',
        help='The expected filename of the annotation aggregation csv.',
    )

    parser.add_argument(
        '-f',
        '--metrics_filename',
        default='metrics.json',
        help='The filename of the metric json.',
    )

    parser.add_argument(
        '--no-meta',
        action='store_true',
        help='Providing this flag witholds the meta information from the metric jsons.',
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Providing this flag will overwrite any preexisting metric files.',
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    for dataset, annotation_aggregation, parent_dir, summary in load(args.input_dir, args.summary_csv, args.annotation_aggregation_csv):
        metric_results = calculate(dataset.df['ground_truth'], annotation_aggregation, dataset.task_type)

        # Save identifying information inside dict for json
        if not args.no_meta:
            metric_results['meta'] = {}
            metric_results['meta']['dataset'] = dataset.dataset
            metric_results['meta']['task_type'] = dataset.task_type
            metric_results['meta']['truth_inference_method'] = summary['truth_inference_method']
            metric_results['meta']['parameters'] = summary['parameters']
            metric_results['meta']['random_seed'] = summary['random_seed']
            metric_results['meta']['datetime'] = datetime.now()

        metric_file = os.path.join(parent_dir, args.metrics_filename)

        if os.path.isfile(metric_file) and not args.overwrite:
            raise FileExistsError('The metric file %s already exists and overwrite flag is not provided. The file will not be overwriten.' % metric_file)

        with open(metric_file, 'w') as results_file:
            json.dumps(metric_results, results_file, indent=4)
