import argparse
from . import utils

# parser = argparse.ArgumentParser(description='Phoneme LM')

# Data
parser = argparse.ArgumentParser()
# Wikipedia
parser.add_argument(
    "--wikipedia-tokenized-file", type=str,
    help="The file in which wikipedia tokenized text should be")
parser.add_argument(
    "--wikipedia-train-file", type=str,
    help="The file in which wikipedia tokenized text should be")
parser.add_argument(
    "--wikipedia-words-file", type=str,
    help="The file in which to annotate target words")
# Embeddings
parser.add_argument(
    "--embeddings-raw-path", type=str,
    help="The file where to store the pickle.bz2 embedding files")
parser.add_argument(
    '--embeddings-merged-path', type=str,
    help='The directory where to save embeddings split per word.')
parser.add_argument(
    '-n', '--n-chars-filesystem', type=int, default=2,
    help='The number of initial characters to use to split words in files.')
# Covariances
parser.add_argument(
    '--embeddings-covariance-path', type=str,
    help='The file where covariances are saved.')
parser.add_argument(
    '--embeddings-variance-file', type=str,
    help='The file where variances are saved.')
parser.add_argument(
    '--min-samples', type=int, default=100,
    help='Minmum number of samples to save covariance.')

# Surprisal
parser.add_argument(
    "--trained-bert-file", type=str,
    help="The file where to store the pickle.bz2 surprisal file")
parser.add_argument(
    '--n-train-files', type=int, default=2,
    help='Number of files used for training surprisal bert.')
parser.add_argument(
    '--min-freq-vocab', type=int, default=5,
    help='Minimum number of times a word needs to appear to be in vocab.')
parser.add_argument(
    "--surprisal-bert-path", type=str,
    help="The path where to store the pickle.bz2 surprisal file")
parser.add_argument(
    "--surprisal-bert-file", type=str,
    help="The file where to store the pickle.bz2 surprisal file")
parser.add_argument(
    "--unigram-probs-file", type=str,
    help="The file where to store the pickle.bz2 surprisal file")
parser.add_argument(
    "--vocab-file", type=str,
    help="The file with trained model real vocab")

# Analysis
parser.add_argument(
    '--language', type=str, default='en',
    help='The language analysed.')
parser.add_argument(
    '--polysemy-variance-file', type=str,
    help='The file where polysemy with variances is saved.')
parser.add_argument(
    '--polysemy-covariance-file', type=str,
    help='The file where polysemy with covariances is saved.')
parser.add_argument(
    "--surprisal-file", type=str,
    help="The file where to store the surprisal tsv")
parser.add_argument(
    '--correlation-variance-file', type=str,
    help='The file where correlation with variances is saved.')
parser.add_argument(
    '--correlation-covariance-file', type=str,
    help='The file where correlation with covariances is saved.')

parser.add_argument(
    "--max-articles", type=int, default=10000, required=False,
    help="The maximum number of articles to be processed")
parser.add_argument(
    "--dump-size", type=int, default=1000, required=False,
    help="The number of articles in each partial dump")
parser.add_argument(
    "--batch-size", type=int, default=40, required=False,
    help="The size of the mini batches")

# Model
parser.add_argument(
    "--bert", default='bert-base-multilingual-cased', required=False,
    help="The name of the pretrained BERT model (default is multilingual)")
parser.add_argument(
    "--spacy", default='xx_ent_wiki_sm', required=False,
    help="The name of the spaCy language model (default is multilingual)")

# Others
parser.add_argument(
    "--cores", type=int, default=8, required=False,
    help="The number of processes to run")
parser.add_argument(
    '--seed', type=int, default=7, required=False,
    help='Seed for random algorithms repeatability (default: 7)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)

    utils.config(args.seed)
    return args
