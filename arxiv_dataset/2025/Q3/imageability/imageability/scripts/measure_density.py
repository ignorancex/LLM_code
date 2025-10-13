from imageability.lib import algorithms, db

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import List
from scipy import stats


parser = argparse.ArgumentParser('measure_density')
parser.add_argument("--output", type=str, help="Path to the output directory")
parser.add_argument("--prefix", type=str, help="Prefix for figures")

parser.add_argument("--algorithm", type=algorithms.Algorithm, choices=list(algorithms.Algorithm), nargs='+')

parser.add_argument("--collections", type=str, nargs="+",
                    help="Path to the collections of embedding vectors")
parser.add_argument("--queries", type=str, help="Path to query embeddings")
parser.add_argument("--gt", type=str, help="Path to the word-score pairs, one line per embedding")
parser.add_argument('--validation', type=float, help='Ratio of queries to use for validation')
parser.add_argument('--seed', type=int, default=43, help='Seed for the random generator')

parser.add_argument("--nn_min", type=int, default=32,
                    help="Minimum number of nearest neighbors to consider for neighborhood analysis")
parser.add_argument("--nn_max", type=int, default=4096,
                    help="Maximum number of nearest neighbors to consider for neighborhood analysis")
parser.add_argument("--nn_step", type=int, default=8,
                    help="Number of nearest neighbors increments")
parser.add_argument("--nn_list", type=int, nargs='+',
                    help="List of radii. Supersedes --nn_min, --nn_max, and --nn_step")

parser.add_argument("--nprobe", type=int, default=128, nargs="?",
                    help="Number of clusters to probe in IVF-style ANN search")
args = parser.parse_args()


def get_nn_list():
    if args.nn_list is not None:
        return args.nn_list
    return range(args.nn_min, args.nn_max, args.nn_step)


def evaluate(algorithm, nn: List[int], index, collection, queries, gt_scores):
    params = algorithms.Params(
        nn = nn,
        nsm_k = 1)

    scores = algorithm.scores(index, collection, queries, params)
    scores = [scores[i, :] for i in range(scores.shape[0])]
    correlations = [algorithm.correlation(gt_scores, nn_scores) for nn_scores in scores]
    return (scores, correlations)


def plot(algorithm, nn, correlation, gt_scores, scores, output):
    print(f' [{algorithm}] Correlation: {correlation}')

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100, constrained_layout=True)
    ax.scatter(gt_scores, scores, s=20, c='deepskyblue')
    ax.set_xlabel(r'\textsc{Ground-truth Score}')
    ax.set_ylabel(r'\textsc{Predicted Score}')
    ax.set_title(r'\textsc{Spearman\'s} $r$: ' + f'{correlation.statistic:.3f}')
    plt.savefig(output / f'{args.prefix}_{algorithm}_{nn}.png')
    plt.close()


def plot_validation_radius(nn_list, correlations, output):
    coefficients = np.array([c.statistic for c in correlations]).tolist()
    with open(output / f'{args.prefix}_validation_radius.txt', 'wt') as f:
        f.write(f'{nn_list}\n')
        f.write(f'{[float(f"{v:0.3}") for v in coefficients]}\n')

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100, constrained_layout=True)
    ax.plot(nn_list, coefficients, linewidth=4, c='darkorange')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'\textsc{Correlation Coefficient}')
    plt.savefig(output / f'{args.prefix}_validation_radius.pdf')
    plt.close()


def plot_validation_scores(nn_list, scores, output):
    if len(nn_list) > 10:
        return

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100, constrained_layout=True)

    colors = ['darkorange', 'deepskyblue', 'maroon', 'black', 'red']
    linestyles = ['-', '--', '-.', ':']
    for i, k in enumerate(nn_list):
        k = int(np.log2(k))
        x = np.linspace(0, 1, 1000)
        kde = stats.gaussian_kde(scores[i])
        ax.plot(x, kde(x),
                linewidth=4,
                linestyle=linestyles[i % len(linestyles)],
                c=colors[i % len(colors)],
                label=r'$2^{'+ str(k) + r'}$')

    ax.set_xlabel(r'\textsc{Nsm}')
    ax.set_ylabel(r'\textsc{Density}')
    plt.legend(fontsize=26)
    plt.savefig(output / f'{args.prefix}_validation_scores.pdf')
    plt.close()


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,bm}'
plt.rcParams.update({'font.size': 36})


output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)


print(r'Loading the data...')
collection = db.load_collection(args.collections)
queries = db.load_collection([args.queries])
gt_words, gt_scores = db.load_groundtruth(args.gt)

if (output / 'faiss_index').exists():
    index = db.index(collection, output / 'faiss_index')
else:
    index = db.index(collection, output / 'faiss_index.bin')

index.nprobe = args.nprobe

rng = np.random.default_rng(seed=args.seed)
validation_count = int(args.validation * queries.shape[0])

nn_list = get_nn_list()

for algorithm in args.algorithm:
    print(f'Running algorithm {algorithm}...')

    if validation_count == 0:
        scores, correlations = evaluate(algorithm, nn_list, index, collection, queries, gt_scores)
        for i, nn in enumerate(nn_list):
            plot(algorithm, nn, correlations[i], gt_scores, scores[i], output)
    else:
        print(f'  Validating on {validation_count} random queries...')
        validation_queries_indices = rng.choice(np.arange(queries.shape[0]), validation_count)
        validation_queries = queries[validation_queries_indices]
        validation_gt_scores = gt_scores[validation_queries_indices]

        eval_queries = np.delete(queries, validation_queries_indices, axis=0)
        eval_gt_scores = np.delete(gt_scores, validation_queries_indices, axis=0)

        scores, correlations = evaluate(algorithm, nn_list, index, collection,
                                   validation_queries, validation_gt_scores)

        plot_validation_radius(nn_list, correlations, output)
        plot_validation_scores(nn_list, scores, output)
        best_nn = nn_list[0]
        best_correlation = 0.
        for i, correlation in enumerate(correlations):
            if correlation.statistic > best_correlation:
                best_correlation = correlation.statistic
                best_nn = nn_list[i]


        print(f'  Testing on the remaining queries with nn={best_nn}')
        scores, correlations = evaluate(algorithm, [best_nn], index,
                                        collection, eval_queries, eval_gt_scores)
        plot(algorithm, best_nn, correlations[0], eval_gt_scores, scores[0], output)

        print(f'  Evaluating on the full set of queries')
        scores, correlations = evaluate(algorithm, [best_nn], index,
                                        collection, queries, gt_scores)
        with open(output / f'{args.prefix}_{algorithm}_scores.csv', 'wt') as f:
            for (word, score) in zip(gt_words, scores[0]):
                f.write(f'{word},{score}\n')
