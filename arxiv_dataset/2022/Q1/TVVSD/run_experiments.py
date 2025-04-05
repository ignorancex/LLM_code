from typing import Tuple

import numpy as np
import pandas as pd

from gtg import gtg, labelling, strategy_space, one_hot, generate_nodes, affinity_matrix, \
    first_sense_strategy_space, mfs_heuristic_strategies
from utils import filter_image_name, combine_data, aggregate_stats

import argparse


def load_data(gold: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, pre-process and combine the required data.

    Args:
        gold: whether to use GOLD annotations or PRED annotations.

    Returns:
        all the features combined, verb senses, sense labels ground truth.
    """

    if gold:
        images_features = pd.read_pickle('data/features/GOLD/images_features_nametrim.pkl')
        embedded_annotations = pd.read_pickle('data/features/GOLD/verse_embedding.pkl')
    else:
        images_features = pd.read_pickle('data/features/PRED/images_features_new.pkl')
        embedded_annotations = pd.read_pickle('data/features/PRED/pred_verse_embedding.pkl')

    images_features['e_image'] = images_features['e_image'].apply(lambda x: x / np.linalg.norm(x, ord=2))
    full_features = combine_data(embedded_annotations, images_features)

    senses = pd.read_csv('data/labels/verse_visualness_labels.tsv', sep='\t', dtype={'sense_num': str})
    sense_labels = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv', dtype={'sense_chosen': str})

    if gold:
        sense_labels['image'] = sense_labels['image'].apply(filter_image_name)
    sense_labels = sense_labels[sense_labels['sense_chosen'] != '-1']  # Drop unclassifiable elements
    sense_labels.reset_index(inplace=True, drop=True)

    return full_features, senses, sense_labels


def infer_senses(ground_truth: pd.DataFrame, affinity: np.ndarray, strategies: np.ndarray, senses: pd.DataFrame,
                 sense_labels: pd.DataFrame, seed: int, labels_per_class: int, feature_type: str, max_it: int,
                 out_fn: str, alpha: float = -1) -> None:
    """
    Run a single GTG experiment.

    Args:
        ground_truth: dataframe of with all data-points ground truths.
        affinity: pair-wise affinity matrix between data features.
        strategies: NxC strategy-space matrix in which each row is a
            verb and each column is a sense.
        senses: A dataframe of verb senses.
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).
        seed: seed used to select a random sample of data-points to label.
        labels_per_class: the maximum number of labeled data-points to
            be sampled in each class.
        out_fn: path of output file containing statistics.
        feature_type: the encoding of data-points features
            (i.e. textual, visual, combined, etc.).
        max_it: the number of RD iterations.
        alpha: current alpha value used for first sense initialisation.
    """
    labels_index = labelling(senses, sense_labels, seed, labels_per_class)
    motions_accuracy, non_motions_accuracy = gtg(ground_truth, affinity, labels_index, strategies, max_it)

    print('\tMotion: {:.3f}'.format(motions_accuracy * 100))
    print('\tNon-motion: {:.3f}'.format(non_motions_accuracy * 100))

    with open(out_fn, 'a+') as out_stream:
        output_str = '{:d},{:.5f},{:s},{:s},{:.4f}\n'
        out_stream.write(output_str.format(labels_per_class, alpha, 'motions', feature_type, motions_accuracy))
        out_stream.write(output_str.format(labels_per_class, alpha, 'non_motions', feature_type, non_motions_accuracy))
        out_stream.close()


def run_gtg_experiment(senses: pd.DataFrame, sense_labels: pd.DataFrame, full_features: pd.DataFrame, max_labels: int,
                       max_it: int, all_senses: bool, fs_init: bool = False, out_fn: str = 'experiments.csv',
                       alpha_min: float = 0.1, alpha_max: float = 1, alpha_step: float = 0.1) -> None:
    """
    Run semi-supervised GTG experiments, they are run with increasing
    elements per label (from 1 to N). Each experiment is replicated 15
    times, each one with a different seed. Results are then written to
    a CSV file.

    Args:
        senses: A dataframe of verb senses.
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).
        full_features: A dataframe containing a different feature
            representation for each column. Each row is a data point.
        max_labels: the maximum number of labeled data-points to
            be sampled in each class, the first run will use 1 single
            labeled data-point per sense. This number will increase
            after each run up to max_labels.
        max_it: the number of RD iterations.
        out_fn: path of output file containing statistics.
        all_senses: if True, the sense probability of a data-point is
            distributed on all senses, otherwise they are restricted to
            the ones related to a given verb.
        fs_init: if True, the sense probability is
            not distributed uniformly between senses, instead the first
            sense is initialised with a given probability alpha value
            and the remaining probability is uniformly distributed
            between other senses. The range of alpha is determined by
            the parameters alpha_min and alpha_max.
        alpha_min: the initial value of alpha (used only in case of
            first sense initialisation).
        alpha_max: the value of alpha after which the experiment will end
            (used only in case of first sense initialisation).
        alpha_step: the amount of probability to sum to alpha after each
            run. It determines the number of runs (used only in case of
            first sense initialisation).
    """
    ground_truth = sense_labels[['lemma', 'sense_chosen']].copy()
    ground_truth['one_hot'] = ground_truth.apply(lambda r: one_hot(senses, r.lemma, r.sense_chosen), axis=1)

    # Uniform distribution on senses related to the verb (Semi-supervised)
    seeds = [73, 37, 29, 30124, 30141, 54321, 1001001, 2051995, 579328629, 1337, 7331, 1221, 111, 99, 666]
    strategies = strategy_space(sense_labels, senses, all_senses)
    for seed in seeds:
        print('\nSeed: {:d}'.format(seed))
        for labels_per_class in range(1, max_labels):
            print('Min labels per class: {:d}'.format(labels_per_class))
            for feature_type in full_features.columns.to_list():
                nodes = generate_nodes(full_features, sense_labels, feature_type)
                affinity = affinity_matrix(nodes)

                if fs_init:
                    for alpha in np.arange(alpha_min, alpha_max + alpha_step, alpha_step):
                        strategies = first_sense_strategy_space(sense_labels, senses, alpha, all_senses)
                        print('\t{:s} alpha: {:.3f}'.format(feature_type, alpha))
                        infer_senses(ground_truth, affinity, strategies, senses, sense_labels,
                                     seed, labels_per_class, feature_type, max_it, out_fn, alpha)
                else:
                    print('{:s}'.format(feature_type))
                    infer_senses(ground_truth, affinity, strategies, senses, sense_labels,
                                 seed, labels_per_class, feature_type, max_it, out_fn)


def first_sense_heuristic(senses: pd.DataFrame, sense_labels: pd.DataFrame, features: pd.DataFrame) -> None:
    print('First Sense heuristic')
    ground_truth = sense_labels[['lemma', 'sense_chosen']].copy()
    ground_truth['one_hot'] = ground_truth.apply(lambda r: one_hot(senses, r.lemma, r.sense_chosen), axis=1)
    nodes = generate_nodes(features, sense_labels, 'e_caption')
    affinity = affinity_matrix(nodes)
    labels_index = np.array([])
    strategies = strategy_space(sense_labels, senses)
    motions_accuracy, non_motions_accuracy = gtg(ground_truth, affinity, labels_index, strategies, 10, True)
    print('\tMotion: {:.3f}'.format(motions_accuracy * 100))
    print('\tNon-motion: {:.3f}'.format(non_motions_accuracy * 100))


def most_frequent_sense_heuristic(senses: pd.DataFrame, sense_labels: pd.DataFrame, features: pd.DataFrame) -> None:
    # Most Frequent Sense Heuristics
    print('Most Frequent Sense')
    # senses = filter_senses(senses, sense_labels)
    y = sense_labels[['lemma', 'sense_chosen']].copy()
    y['one_hot'] = y.apply(lambda r: one_hot(senses, r.lemma, r.sense_chosen), axis=1)
    nodes = generate_nodes(features, sense_labels, 'e_object')
    affinity = affinity_matrix(nodes)
    strategies = mfs_heuristic_strategies(sense_labels, senses)
    labels_index = np.array([])
    motions_accuracy, non_motions_accuracy = gtg(y, affinity, labels_index, strategies, 10, True)
    print('\tMotion: {:.3f}'.format(motions_accuracy * 100))
    print('\tNon-motion: {:.3f}'.format(non_motions_accuracy * 100))


def main():
    """ Run visual verb disambiguation experiments. """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-G', '--gold_captions', action='store_const', const='gold', default='gold', dest='captions',
                       help='Use GOLD captions from COCO.')
    group.add_argument('-P', '--pred_captions', action='store_const', const='pred', dest='captions',
                       help='use PRED captions extracted with NeuralBabyTalk.')
    parser.add_argument('-m', '--max_labels', type=int, default=20,
                        help='The maximum number of labeled data-points to use for each sense.')
    parser.add_argument('-i', '--iterations', type=int, help='The number of Replicator Dynamics iterations to be run.')
    parser.add_argument('-a', '--all_senses', action='store_true',
                        help='Ignore input verb, run inference on the senses of all verbs for each data point.')

    args = parser.parse_args()
    use_gold = args.captions == 'gold'
    use_all_senses = args.all_senses
    max_labels = args.max_labels

    if args.iterations:
        max_it = args.iterations
    else:
        max_it = 100

    features, senses, sense_labels = load_data(use_gold)
    run_gtg_experiment(senses, sense_labels, features, max_labels, max_it, use_all_senses)
    aggregate_stats('experiments.csv')


if __name__ == '__main__':
    main()
