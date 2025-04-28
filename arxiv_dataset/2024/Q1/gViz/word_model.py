import os
import sys
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors


class WordModel(object):
    def __init__(self, model_path):
        self.model_path = model_path

        self.vector_size = 256
        self.window = 3
        self.min_count = 1
        self.worker = 4
        self.epochs = 10
        self.sg = 1
        self.negative = 20

        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            return None
        else:
            model = KeyedVectors.load(self.model_path)
            print("# terms of word model: {:,}".format(len(model.vocab)))
            return model

    def train(self, corpus):
        model = Word2Vec(sentences=corpus, size=self.vector_size, window=self.window,
                         min_count=self.min_count, workers=self.worker, sg=self.sg, negative=self.negative,
                         iter=self.epochs)
        model.wv.save(self.model_path)

    def nearest_terms(self, term, n=10):
        if term in self.model.vocab:
            res = self.model.most_similar(term, topn=n)
            return [x[0] for x in res]
        return []


def _build_corpus(data_path):
    df = pd.read_excel(data_path)
    corpus = df["title"].tolist() + df["abstract"].tolist() + df["keywords"].tolist()
    return [x.split(",") for x in corpus if isinstance(x, str)]


def main():
    if len(sys.argv) >= 3:
        step, data_dir = sys.argv[1], sys.argv[2]
    else:
        print("python {} [step: train|test] [data_dir]".format(sys.argv[0]))

    model_path = os.path.join(data_dir, "word.model")
    word_model = WordModel(model_path)

    if step == "train":
        data_path = os.path.join(data_dir, "dataset.xlsx")
        corpus = _build_corpus(data_path)
        word_model.train(corpus)
    elif step == "test":
        term = sys.argv[3]
        rv = word_model.nearest_terms(term)
        print(rv)


if __name__ == "__main__":
    main()
