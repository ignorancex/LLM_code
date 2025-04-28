import os
import sys
import pandas as pd

from nlp import clean, word_tokenize, pos_tagging, stemming, filtering


class Preprocess(object):
    def __init__(self, data_path, out_dir):
        print("load {}".format(__class__.__name__))
        self.data_path = data_path
        self.out_dir = out_dir

        self.df = self._load_data()

    def _load_data(self):
        return pd.read_excel(self.data_path)

    def _get_terms(self, text):
        if not isinstance(text, str):
            return None

        text = clean(text)
        tokens = word_tokenize(text)
        nouns = pos_tagging(tokens)
        stemmed = stemming(nouns)
        filtered = filtering(stemmed)
        return ",".join(filtered) if filtered else None

    def preprocess(self):
        print("# raw data: {:,}".format(self.df.shape[0]))
        df = self.df
        df["keywords"] = df["keywords_raw"].map(lambda x: self._get_terms(x))
        df["abstract"] = df["abstract_raw"].map(lambda x: self._get_terms(x))
        df["title"] = df["title_raw"].map(lambda x: self._get_terms(x))
        self.df = df[~df.keywords.isna() & ~df.abstract.isna()]
        print("# preprocessed data: {:,}".format(self.df.shape[0]))

    def save(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        save_path = os.path.join(self.out_dir, "dataset.xlsx")
        self.df.to_excel(save_path)
        print("save {} done.".format(save_path))

    def run(self):
        self.preprocess()
        self.save()


def main():
    if len(sys.argv) == 3:
        data_path, out_dir = sys.argv[1], sys.argv[2]
    else:
        print("python {} [data_path: *.xlsx] [out_dir]".format(sys.argv[0]))
        sys.exit(1)

    prep = Preprocess(data_path, out_dir)
    print("initialization done.")

    prep.run()
    print("finish!")


if __name__ == "__main__":
    main()
