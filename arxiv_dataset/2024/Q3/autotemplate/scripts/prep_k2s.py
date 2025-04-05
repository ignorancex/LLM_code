import json
import random
import string
from pathlib import Path

import fire
import nltk
from tqdm import tqdm, trange


def train(data_dir, trials: int = 2):
    stopwords = set(nltk.corpus.stopwords.words('english')) | set(string.punctuation)
    for split in ("train", "val"):
        with open(data_dir / f"{split}.jsonl", "w") as file:
            if split == "val":
                split = "dev"
            for line in tqdm(list(open(data_dir / f"{split}_v2.txt")), desc=str(data_dir / split)):
                words = line.split()
                index = [i for w, i in zip(words, list(range(len(words)))) if
                         w.lower() not in stopwords and w[0].islower()]
                for _ in range(trials):
                    for k in range(1, 7):
                        if len(index) < k:
                            continue
                        src_words = sorted(random.sample(index, k=k))
                        src, tgt = "TL;DR:<extra_id_0>", "<extra_id_0>"
                        i = 1
                        prev_word_idx = -1
                        for word_idx in src_words:
                            src += f" {words[word_idx]}<extra_id_{i}>"

                            if prev_word_idx + 1 != word_idx:
                                tgt += " " + " ".join(words[prev_word_idx + 1:word_idx])
                            tgt += f"<extra_id_{i}>"
                            prev_word_idx = word_idx
                            i += 1
                        if len(words[prev_word_idx + 1:]):
                            tgt += " " + " ".join(words[prev_word_idx + 1:])
                        tgt += f"<extra_id_{i}>"
                        print(json.dumps({"src": src, "tgt": tgt, "ref": line.strip()}), file=file)


def test(data_dir):
    data_dir = Path(data_dir)
    for i in trange(1, 7, desc=str(data_dir)):
        with open(data_dir / f"{i}keywords.jsonl", "w") as file, open(data_dir / f"{i}keywords.txt") as src_file:
            while True:
                try:
                    _ = next(src_file)
                    keywords = next(src_file).split("\t")[1].split()
                    ref = next(src_file).split("\t")[1].strip()
                    src = "TL;DR:<extra_id_0>"
                    extra_id = 1
                    for word in keywords:
                        src += " " + word + f"<extra_id_{extra_id}>"
                        extra_id += 1
                    print(json.dumps({"src": src, "ref": ref, "keywords": keywords}), file=file)
                except StopIteration:
                    break


def process(data_dir: Path):
    train(data_dir)
    test(data_dir)


def run(data_dir="./data/lm"):
    data_dir = Path(data_dir)
    process(data_dir / "one-billion-words")
    process(data_dir / "yelp_review")


if __name__ == '__main__':
    fire.Fire(run)
