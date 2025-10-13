#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = (
    "evaluate Ada Boost Classifier "
)
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import argparse
import joblib
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.decomposition import PCA

# BioBERT Model Identifier
BERT_MODEL_IDENTIFIER = "dmis-lab/biobert-v1.1"
max_length = 512


class EvalDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }


def load_evaldata(evaldata):
    """Load evaluation data from CSV."""
    df = pd.read_csv(evaldata).dropna()
    texts = df["text"].tolist()
    labels = df["category_id"].tolist()
    print(f"Loaded {len(texts)} samples.")
    return texts, labels


def tokenize_and_embed(texts):
    """Tokenizes text and extracts BioBERT embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_IDENTIFIER)
    model = AutoModel.from_pretrained(BERT_MODEL_IDENTIFIER)

    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            output = model(**tokens)
        embeddings.append(
            output.last_hidden_state[:, 0, :].squeeze().numpy()
        )  # CLS token
    embeddings = np.array(embeddings)
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)

    return reduced_embeddings


def convert_labels(labels):
    """Converts labels to one-hot encoding."""
    label_map = {"scientific": 0, "popular": 1, "disinfo": 2, "alternative_science": 3}
    labels_conv = [label_map[label] for label in labels]
    return np.array(labels_conv)


def evaluate_model(classifier, embeddings, labels):
    """Evaluates the AdaBoost classifier."""
    predictions = classifier.predict(embeddings)
    f1 = f1_score(labels, predictions, average="weighted")
    class_report = classification_report(
        labels,
        predictions,
        target_names=["scientific", "popular", "disinfo", "alternative_science"],
    )
    return f1, class_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the trained AdaBoost model")
    parser.add_argument("evaldata", help="Path to the evaluation dataset (CSV)")
    args = parser.parse_args()

    classifier = joblib.load(args.model)
    texts, labels = load_evaldata(args.evaldata)

    reduced_embeddings = tokenize_and_embed(texts)
    labels = convert_labels(labels)

    f1, class_report = evaluate_model(classifier, reduced_embeddings, labels)

    print(f"F1 Score: {f1}")
    print("Classification Report:\n", class_report)


if __name__ == "__main__":
    main()
