from imageability.lib import db

from tqdm import tqdm
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE


plt.rcParams.update({'font.size': 34})

parser = argparse.ArgumentParser('visualize')
parser.add_argument("--output", type=str, help="Path to the output directory")

parser.add_argument("--collections", type=str, nargs="+",
                    help="Path to the collections of embedding vectors")
parser.add_argument("--queries", type=str, help="Path to query embeddings")
parser.add_argument("--gt", type=str,
                    help="Path to the word-score pairs, one line per embedding")

parser.add_argument("--imageable_words", type=str, nargs="+",
                    help="List of example imageable words",
                    default=['banana', 'beach', 'skunk', 'jeep', 'beetle'])
parser.add_argument("--abstract_words", type=str, nargs="+",
                    help="List of example abstract words",
                    default=['legality', 'circumstance', 'any', 'whether'])

parser.add_argument("--nn", type=int, default=1024, nargs="?",
                    help="Maximum number of nearest neighbors to consider for neighborhood analysis")

parser.add_argument("--nprobe", type=int, default=128, nargs="?",
                    help="Number of clusters to probe in IVF-style ANN search",)
args = parser.parse_args()


def plot_tsne(vectors: np.ndarray,
              imageable_words,
              imageable_embeddings: np.ndarray,
              abstract_words,
              abstract_embeddings: np.ndarray,
              path: Path):
    tsne = TSNE(n_components=2, metric='cosine', init='pca')

    collection = [vectors]
    if imageable_words is not None:
        collection.append(imageable_embeddings)
    if abstract_words is not None:
        collection.append(abstract_embeddings)

    print('Executing TSNE...')
    transformed = tsne.fit_transform(np.concatenate(collection, axis=0))

    print('Plotting...')
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100, constrained_layout=True)

    begin, end = 0, vectors.shape[0]
    plt.scatter(transformed[begin:end, 0], transformed[begin:end, 1], c='silver', s=10)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_axis_off()

    if imageable_words is not None:
        begin, end = end, end + len(imageable_words)
        plt.scatter(transformed[begin:end, 0], transformed[begin:end, 1], c='royalblue', hatch='/', s=200)
        for i, word in enumerate(imageable_words):
            plt.annotate(word, (transformed[begin+i, 0], transformed[begin+i, 1] + 2.5), c='blue', ha='center', fontsize=36)

    if abstract_words is not None:
        begin, end = end, end + len(abstract_words)
        plt.scatter(transformed[begin:end, 0], transformed[begin:end, 1], c='maroon', hatch='+', s=200)
        for i, word in enumerate(abstract_words):
            plt.annotate(word, (transformed[begin+i, 0], transformed[begin+i, 1] + 2.5), c='red', ha='center', fontsize=36)

    plt.savefig(path)


output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)


print(r'Loading the data...')
collection = db.load_collection(args.collections)
queries = db.load_collection([args.queries])
gt_words, gt_scores = db.load_groundtruth(args.gt)
index = db.index(collection, output / 'faiss_index')
index.nprobe = args.nprobe


print('Relevant statistics:')
print(f'  - Number of query words: {len(gt_words)}')
print(f'  - Number of vectors in collection: {collection.shape[0]}')


visualized_points = []
imageables = []
imageable_words = []
abstracts = []
abstract_words = []

_, neighborhoods = index.search(queries, args.nn)

for qid, neighborhood in enumerate(tqdm(neighborhoods)):
    if gt_words[qid] in args.imageable_words:
        visualized_points.append(collection[neighborhood])
        imageables.append(queries[qid])
        imageable_words.append(gt_words[qid])
    if gt_words[qid] in args.abstract_words:
        visualized_points.append(collection[neighborhood])
        abstracts.append(queries[qid])
        abstract_words.append(gt_words[qid])

visualized_points = np.concatenate(visualized_points, axis=0)
plot_tsne(visualized_points, imageable_words, imageables, abstract_words, abstracts, output / f'tsne.pdf')
