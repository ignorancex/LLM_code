import os
import sys
import pandas as pd
from collections import Counter

from word_model import WordModel


class MakeCorpus(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.min_freq = 3

        self.word_model = WordModel(os.path.join(self.data_dir, "word.model"))

    def run(self):
        df = self._load_dataset()
        print("# data: {:,}".format(df.shape[0]))

        abs_terms, kwd_terms = self._load_terms(df)
        print("# unique terms: abstract - {:,}, keywords - {:,}".format(len(abs_terms), len(kwd_terms)))

        nearest_terms = self._get_nearest_terms(kwd_terms)
        print("# nearest terms: {:,}".format(len(nearest_terms)))

        vocab = self._build_vocab(abs_terms, nearest_terms)
        print("build vocab done.")

        self._build_corpus(df, vocab)
        print("build corpus done.")

    def _load_dataset(self):
        return pd.read_excel(os.path.join(self.data_dir, "dataset.xlsx"))

    def _load_terms(self, df):
        total_abs_terms, total_kwd_terms = list(), list()
        for d in df[["abstract", "keywords"]].values:
            total_abs_terms.extend(d[0].split(","))
            total_kwd_terms.extend(d[1].split(","))
        print("# total terms: abstract - {:,}, keywords - {:,}".format(len(total_abs_terms), len(total_kwd_terms)))

        abs_terms = [term for term, freq in Counter(total_abs_terms).items() if freq >= self.min_freq]
        kwd_terms = [term for term, freq in Counter(total_kwd_terms).items() if freq >= self.min_freq]
        return abs_terms, kwd_terms

    def _get_nearest_terms(self, vocab):
        nearest_terms = set()
        for term in vocab:
            terms = self.word_model.nearest_terms(term)
            if terms:
                nearest_terms.update(terms)
        return nearest_terms

    def _build_vocab(self, abs_terms, nearest_terms):
        terms = set(abs_terms) & set(nearest_terms)
        vocab_path = os.path.join(self.data_dir, "vocab.dat")
        vocab = dict()
        with open(vocab_path, "w") as f:
            for idx, term in enumerate(terms):
                vocab[term] = idx
                f.write("{}\t{}\n".format(idx, term))
        print("# vocab: {:,} (save path: {})".format(len(terms), vocab_path))
        return vocab

    def _build_corpus(self, df, vocab):
        corpus_path = os.path.join(self.data_dir, "corpus.dat")
        vocab_set = set(vocab.keys())
        n_corpus = 0
        with open(corpus_path, "w") as f:
            for d in df["abstract"]:
                terms = set(d.split(",")) & vocab_set
                if terms:
                    f.write(" ".join(map(str, [vocab[t] for t in terms])) + "\n")
                    n_corpus += 1
        print("# corpus: {:,} (save path: {})".format(n_corpus, corpus_path))


def main():
    if len(sys.argv) == 2:
        data_dir = sys.argv[1]
    else:
        print("python {} [data_dir]".format(sys.argv[0]))

    x = MakeCorpus(data_dir)
    print("initialization done.")

    x.run()
    print("finish!")


if __name__ == "__main__":
    main()
