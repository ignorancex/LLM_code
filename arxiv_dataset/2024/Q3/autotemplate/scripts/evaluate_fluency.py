from collections import defaultdict

import fire
import nltk
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline


def run(file_path: str,
        model_name: str = "cointegrated/roberta-large-cola-krishna2020"):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model_name, device=device)
    scores = defaultdict(list)
    for line in tqdm(list(open(file_path))):
        sentences = nltk.sent_tokenize(line.strip())
        prob = [x["score"] for x in pipe(sentences)]
        scores["micro"].extend(prob)
        scores["macro"].append(sum(prob) / len(prob))
        if len(scores["macro"]) % 500 == 0:
            print({k: sum(v) / len(v) for k, v in scores.items()})
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    print(pd.Series(scores))


if __name__ == '__main__':
    fire.Fire(run)
