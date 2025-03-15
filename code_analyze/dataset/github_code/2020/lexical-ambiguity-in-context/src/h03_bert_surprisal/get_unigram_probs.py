# -*- coding: utf-8 -*-
import sys
import argparse
import math
import pickle
from tqdm import tqdm

sys.path.append('./src/')
from utils import argparser as parser
from utils import utils


def process_sentence(tokens, frequencies):
    for word in tokens:
        if word in frequencies:
            frequencies[word] += 1
    return frequencies


def process_file(filename, words):
    frequencies = {word: 0 for word in words}
    n_tokens = 0

    with open(filename, 'r') as f:
        for line in tqdm(f, desc='Processing wikipedia'):
            tokens = line.strip().split(' ')
            frequencies = process_sentence(tokens, frequencies)
            n_tokens += len(tokens)

    return frequencies, n_tokens


def get_frequencies(words, filename):
    print('Getting frequencies for %d words' % (len(words)))

    frequencies, n_tokens = process_file(filename, words)
    return frequencies, n_tokens


def get_probabilities(frequencies, n_tokens):
    probs = {word: ((count + 1) / (n_tokens + len(frequencies))) for word, count in frequencies.items()}
    surprisals = {word: - math.log(prob, 2) for word, prob in probs.items()}
    return probs, surprisals


def save_frequencies(fname, probs, surprisals, frequencies, n_tokens):
    data = {
        'surprisals': surprisals,
        'probabilities': probs,
        'frequencies': frequencies,
        'n_tokens': n_tokens,
    }
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_surprisals(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data['surprisals']


def load_frequencies(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data['frequencies'], data['n_tokens']


def main():
    args = parser.parse_args()
    tgt_words = utils.read_pickle(args.wikipedia_words_file)

    frequencies, n_tokens = get_frequencies(tgt_words, args.wikipedia_tokenized_file)
    probs, surprisals = get_probabilities(frequencies, n_tokens)
    save_frequencies(args.unigram_probs_file, probs, surprisals, frequencies, n_tokens)


if __name__ == '__main__':
    main()
