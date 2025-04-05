"""
The baseine metrics used to compare two (or multiple) random variables to one
another, providing an (ideally) informative and interpretable distance.

To be mathematically correct, some of these are measures, that provide the size
or quantity of a set, rather than a distance (such as Mutual Information)
"""
import csv
from math import sqrt
import os
import json

import numpy as np
from scipy.stats import multinomial, entropy, wasserstein_distance
from sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt

from psych_metric.datasets import data_handler

def load(input_dir, summary_csv='summary.csv', annotation_aggregation_csv='annoatation_aggregation.csv'):
    """Traverses the given root directory to find directories that contain a
    summary.csv, loads the dataset with ground truth based on dataset_id, and
    compares that ground truth to the contents within the annotation
    aggregation. The calculated metrics are then saved in the same

    Parameters
    ----------
    input_dir : str
        The filepath to the input directory to be traversed to find directories
        that contain the results of annotation aggregation methods.
    summary : str, optional
        The filename of the summary csv
    annotation_aggregation : str, optional
        The filename of the annotation aggregation output.

    Yields
    ------
        dataset class object, annotation_aggregation pd.DataFrame, parent_dir
        str, and summary dict
    """
    # NOTE, this may make more sense as an outside of package script, keeping only calculate() in here.
    dir_queue = [input_dir]

    # Traverse through the directory tree with input_dir as root dir
    while dir_queue:
        focus_dir = dir_queue.pop()
        dir_files = [os.path.join(focus_dir, dir_item) for dir_item in os.listdir(focus_dir)]

        # needs both the corresponding summary.csv and annotation_aggregation.csv
        summary_file = None
        annotation_aggregation_file = None

        # Check for summary and annotation aggregation csvs; add dirs to queue
        while dir_files:
            focus_file = dir_files.pop()

            if os.is_file(focus_file):
                focus_file_end = focus_file.rsplit(os.path.sep)[-1]

                if summary_csv == focus_file_end:
                    summary_file = focus_file

                elif annotation_aggregation_csv == focus_file_end:
                    annotation_aggregation_file = focus_file

            elif os.is_dir(focus_file):
                dir_queue = [focus_file] + dir_queue

        if summary_file is not None and annotation_aggregation_file is not None:
            with open(summary_file, 'r') as summary_fp:
                reader = csv.reader(summary_fp)
                summary = {row[0]:row[1] for rows in reader}

            yield data_handler.load_dataset(summary['dataset'], ground_truth=True), pd.read_csv(annotation_aggregation_file), focus_dir, summary

def calculate(target, predictions, task_type=None, metrics=None):
    """Function handler that calls the appropriate metrics on the given data.

    Parameters
    ----------
    target : array-like
        The target values the predictions are compared to.
    predictions : array-like
        The prediction values to be compared to the target values.
    task_type : str, optional
        The task type performed on the data. If not provided, only task type
        agnostic metrics will be used, if any apply.

    metrics : set(str), optional
        A set of metric identifiers to be calculated if appropriate to the data
        and task type. By default, attempts to calcualte all possible metrics.

    Returns
    -------
        dict of the metrics and their calculations on the data.
    """
    metric_results = dict()

    # TODO need to check dimensions of target and predictions and include in conditionals.

    # TODO could make this way more efficient since there is a lot of orverlapping parts among these metrics.

    # task_type agnostic metrics:
    if metrics is None or 'mutual_information' in metrics or 'kl_divergence' in metrics:
        # NOTE confirm mutual information == kl_divergence
        metric_results['mutual_information'] = entropy(target, predictions)

    #if metrics is None or 'variational_information' in metrics:
    #    # the metric version of mutual information.

    if metrics is None or 'wasserstein' in metrics or 'earth_movers_distance' in metrics:
        metric_results['wasserstein'] = wasserstein_distance(target, predictions)

    if task_type == 'regression':
        if metrics is None or'mean_absolute_error' in metrics:
            metric_results['mean_absolute_error'] = skl_metrics.mean_absolute_error(target, predictions)
        if metrics is None or 'mse' in metrics or 'mean_squared_error' in metrics:
            metric_results['mean_squared_error'] = skl_metrics.mean_squared_error(target, predictions)

        if metrics is None or 'rmse' in metrics:
            metric_results['root_mean_squared_error'] = math.sqrt(skl_metrics.mean_squared_error(target, predictions))

        if metrics is None or'median_absolute_error' in metrics:
            metric_results['median_absolute_error'] = skl_metrics.median_absolute_error(target, predictions)

        if metrics is None or 'r_squared' in metrics:
            metric_results['r_squared'] = skl_metrics.r2_score(target, predictions)

    # TODO require standardized format of the predictions and target to be able to infer the correct metric to use (ie. a sample is a scalar to scalar comparison, or distribution to distribution, etc.)


    elif 'classification' in task_type:
        # TODO handle the multi classification cases if they are not already handled by sklearn
        if metrics is None or 'confusion_matrix' in metrics:
            if 'binary' in task_type
                metric_results['confusion_matrix'] = skl_metrics.confusion_matrix(target, predictions)
            else:

                metric_results['confusion_matrix'] = skl_metrics.multilabel_confusion_matrix(target, predictions)

        #if metrics is None or 'specificity' in metrics:
        #    metric_results['specificity'] = skl_metrics.specificity(target, predictions)

        #if metrics is None or 'sensitivity' in metrics:
        #    metric_results['sensitivity'] = skl_metrics.sensitivity(target, predictions)

        if metrics is None or 'accuracy' in metrics:
            metric_results['accuracy'] = skl_metrics.accuracy_score(target, predictions)

        if metrics is None or 'precision' in metrics:
            metric_results['precision'] = skl_metrics.precision_score(target, predictions)

        if metrics is None or 'recall' in metrics:
            metric_results['recall'] = skl_metrics.recall_score(target, predictions)

        if metrics is None or 'f1' in metrics:
            metric_results['f1'] = skl_metrics.f1_score(target, predictions)

        #if metrics is None or 'roc' in metrics: # area under curve
        #    #metric_results['accuracy'] = skl_metrics.recall_score(target, predictions)
        #    # TODO use the roc class.

        #if metrics is None or 'toc' in metrics: # area under curve


    #elif 'ordering' in task_type: # ranking

    #elif 'mapping' in task_type:

    return metric_results


def parse_args():
    """Parses the arguments when this script is called.

    Returns
    -------
    argparse.Namespace
        The configuration variables for the experiments to be run.
    """
    parser = argparse.ArgumentParser(description='Traverse the directory tree of the given directory and calculates the desired metrics of the saved annotation aggregation csv when compared to the ground truth of the dataset.')

    parser.add_argument('input_dir', default=None, help='Enter the file path to the root input directory.')
    parser.add_argument('-s', '--summary_csv', default='summary.csv', help='The expected filename of the summary csv.')
    parser.add_argument('-a', '--annotation_aggregation_csv', default='annotation_aggregation.csv', help='The expected filename of the annotation aggregation csv.')
    parser.add_argument('-m', '--metrics', default=None, nargs='+', help='List of the metrics to use.')
    parser.add_argument('-f', '--metrics_filename', default='metrics.json', help='The filename of the metric json.')
    parser.add_argument('--no-meta', action='store_true', help='Providing this flag witholds the meta information from the metric jsons.')
    parser.add_argument('--overwrite', action='store_true', help='Providing this flag will overwrite any preexisting metric files.')

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
            json.dump(metric_results, results_file, indent=4)
