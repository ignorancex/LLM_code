import json
from collections import defaultdict
from pathlib import Path

import fire
import nltk
import pandas as pd


def run(reference_dir, hypothesis_dir):
    reference_dir = Path(reference_dir)
    hypothesis_dir = Path(hypothesis_dir)
    # BLEU-2/4, NIST-2/4, METEOR
    scores = defaultdict(list)
    for key in ("beam", "greedy", "sample"):
        if key != "beam":  # Debug
            continue
        for i in range(1, 7):
            if key == "beam":
                hyp_path = hypothesis_dir / f"{i}keywords.txt"
            else:
                hyp_path = hypothesis_dir / f"{i}keywords_{key}.txt"
            hyp = [nltk.word_tokenize(x.strip()) for x in open(hyp_path)]
            scores[f"len_{key}"].append(sum(map(len, hyp)) / len(hyp))
            ref_data = [json.loads(x) for x in open(reference_dir / f"{i}keywords.jsonl")]
            ref = [nltk.word_tokenize(x["ref"]) for x in ref_data]
            keywords = [x["keywords"] for x in ref_data]
            scores[f"coverage_{key}"].append(100 * sum(all(k in " ".join(h) for k in ks) for h, ks in zip(hyp, keywords)) / len(hyp))

            for h, ks in zip(hyp, keywords):
                if all(k in " ".join(h) for k in ks) != 1.:
                    print(ks)
                    print(h)
                    print()

            bleu2 = nltk.translate.bleu_score.corpus_bleu([[r] for r in ref], hyp, weights=(0.5, 0.5, 0., 0.))
            bleu4 = nltk.translate.bleu_score.corpus_bleu([[r] for r in ref], hyp)
            scores[f"bleu-2_{key}"].append(100 * bleu2)
            scores[f"bleu-4_{key}"].append(100 * bleu4)
            scores[f"nist-2_{key}"].append(nltk.translate.nist_score.corpus_nist([[r] for r in ref], hyp, n=2))
            scores[f"nist-4_{key}"].append(nltk.translate.nist_score.corpus_nist([[r] for r in ref], hyp, n=4))
        #     meteor = [nltk.translate.meteor([r], h) for r, h in zip(ref, hyp)]
        #     scores["meteor"].append(sum(meteor) / len(meteor))

    scores = {key: float(f"{sum(scores[key]) / len(scores[key]):.2f}") for key in scores}
    json.dump(scores, open(hypothesis_dir / "score_avg.json", "w"))

    print(hypothesis_dir)
    print(pd.Series(scores))


if __name__ == '__main__':
    fire.Fire(run)
