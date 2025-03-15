# coding=utf-8
# Copyright 2021 The Rliable Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aggregate Performance Estimators."""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties


def aggregate_mean(scores: np.ndarray):
  """Computes mean of sample mean scores per task.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
  Returns:
    Mean of sample means.
  """
  mean_task_scores = np.mean(scores, axis=0, keepdims=False)
  return np.mean(mean_task_scores, axis=0)


def aggregate_median(scores: np.ndarray):
  """Computes median of sample mean scores per task.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
  Returns:
    Median of sample means.
  """
  mean_task_scores = np.mean(scores, axis=0, keepdims=False)
  return np.median(mean_task_scores, axis=0)


def aggregate_optimality_gap(scores: np.ndarray, gamma=1):
  """Computes optimality gap across all runs and tasks.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
    gamma: Threshold for optimality gap. All scores above `gamma` are clipped
     to `gamma`.

  Returns:
    Optimality gap at threshold `gamma`.
  """
  return gamma - np.mean(np.minimum(scores, gamma))


def aggregate_iqm(scores: np.ndarray):
  """Computes the interquartile mean across runs and tasks.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
  Returns:
    IQM (25% trimmed mean) of scores.
  """
  return scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=None)


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray):
  """Overall Probability of imporvement of algorithm `X` over `Y`.

  Args:
    scores_x: A matrix of size (`num_runs_x` x `num_tasks`) where scores_x[n][m]
      represent the score on run `n` of task `m` for algorithm `X`.
    scores_y: A matrix of size (`num_runs_y` x `num_tasks`) where scores_x[n][m]
      represent the score on run `n` of task `m` for algorithm `Y`.
  Returns:
      P(X_m > Y_m) averaged across tasks.
  """
  num_tasks = scores_x.shape[1]
  task_improvement_probabilities = []
  num_runs_x, num_runs_y = scores_x.shape[0], scores_y.shape[0]
  for task in range(num_tasks):
    if np.array_equal(scores_x[:, task], scores_y[:, task]):
      task_improvement_prob = 0.5
    else:
      task_improvement_prob, _ = scipy.stats.mannwhitneyu(
          scores_x[:, task], scores_y[:, task], alternative='greater')
      task_improvement_prob /= (num_runs_x * num_runs_y)
    task_improvement_probabilities.append(task_improvement_prob)
  return np.mean(task_improvement_probabilities)


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize='large'):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))
    return ax


def plot_interval_estimates(point_estimates,
                            interval_estimates,
                            metric_names,
                            algorithms=None,
                            colors=None,
                            color_palette='colorblind',
                            max_ticks=4,
                            subfigure_width=3.4,
                            row_height=0.37,
                            xlabel_y_coordinate=-0.1,
                            xlabel='Normalized Score',
                            bold_best=False,
                            dpi=100,
                            **kwargs):
    """Plots various metrics with confidence intervals.
    Args:
      point_estimates: Dictionary mapping algorithm to a list or array of point
        estimates of the metrics to plot.
      interval_estimates: Dictionary mapping algorithms to interval estimates
        corresponding to the `point_estimates`. Typically, consists of stratified
        bootstrap CIs.
      metric_names: Names of the metrics corresponding to `point_estimates`.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Maps each method to a color. If None, then this mapping is created
        based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
        `plt.MaxNLocator`.
      subfigure_width: Width of each subfigure.
      row_height: Height of each row in a subfigure.
      xlabel_y_coordinate: y-coordinate of the x-axis label.
      xlabel: Label for the x-axis.
      bold_best: If True, bolds the best performing algorithm in each metric.
      **kwargs: Arbitrary keyword arguments.
    Returns:
      fig: A matplotlib Figure.
      axes: `axes.Axes` or array of Axes.
    """

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    num_metrics = len(point_estimates[algorithms[0]])

    figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize, dpi=dpi)
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))
    h = kwargs.pop('interval_height', 0.6)
    if bold_best: # Find the best algorithm using only the first metric.
        best_algorithm = algorithms[np.argmax([point_estimates[alg][0] for alg in algorithms])]
        print(f"Best algorithm: {best_algorithm}")

    for idx, metric_name in enumerate(metric_names):
        for alg_idx, algorithm in enumerate(algorithms):
            ax = axes[idx] if num_metrics > 1 else axes
            # Plot interval estimates.
            lower, upper = interval_estimates[algorithm][:, idx]
            ax.barh(y=alg_idx,
                    width=upper - lower,
                    height=h,
                    left=lower,
                    color=colors[algorithm],
                    alpha=0.75,
                    label=algorithm)
            # Plot point estimates.
            ax.vlines(x=point_estimates[algorithm][idx],
                      ymin=alg_idx - (7.5 * h / 16),
                      ymax=alg_idx + (6 * h / 16),
                      label=algorithm,
                      color='k',
                      alpha=0.5)

        ax.set_yticks(list(range(len(algorithms))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(algorithms, fontsize='x-large')
        ax.set_title(metric_name, fontsize='xx-large')
        ax.tick_params(axis='both', which='major')
        _decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
        for label, alg in zip(ax.get_yticklabels(), algorithms):
            if alg == best_algorithm:
                label.set_fontproperties(FontProperties(weight='semibold', size='xx-large'))
            else:
                label.set_fontproperties(FontProperties(weight='normal', size='xx-large'))
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.25)

    fig.text(0.4, xlabel_y_coordinate, xlabel, ha='center', fontsize='xx-large')
    plt.subplots_adjust(wspace=kwargs.pop('wspace', 0.11), left=0.0)
    return fig, axes