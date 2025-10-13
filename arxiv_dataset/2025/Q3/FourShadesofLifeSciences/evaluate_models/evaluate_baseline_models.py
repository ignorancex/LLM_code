# -*- coding: utf-8 -*-

__description__ = "request non pretrained models"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import argparse
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.cluster import KMeans

BERT_MODEL_IDENTIFIER = "mistralai/Mistral-7B-v0.3" # change here for different model


class EvalDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)  # One-hot encoded

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }


# Load pre-trained BioBERT model and tokenizer
def load_model_tokenizer(BERT_MODEL_IDENTIFIER):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_IDENTIFIER)
    model = BertModel.from_pretrained(BERT_MODEL_IDENTIFIER)
    model.eval()
    return tokenizer, model


def load_testdata(dataset):
    df = pd.read_csv(dataset)
    df = df.dropna()

    texts = df["text"].to_list()
    labels = df["category_id"].to_list()
    print("testdata loaded")
    return texts, labels


def convert_labels(labels):
    label_map = {"scientific": 0, "popular": 1, "disinfo": 2, "alternative_science": 3}
    labels_conv = [label_map[label] for label in labels]
    labels_conv = torch.tensor(labels_conv, dtype=torch.long)
    true_labels = torch.nn.functional.one_hot(labels_conv, num_classes=4).float()
    print("labels converted")
    return true_labels


def prepare_dataloader(text_tokens, labels, batch_size=1):
    dataset = EvalDataset(
        text_tokens["input_ids"], text_tokens["attention_mask"], labels
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("dataloader prepared")
    return dataloader


def generate_embeddings(tokenizer, model, texts):
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # last layer of the model and calculating mean of them for mean pooling
    embeddings = outputs.last_hidden_state.masked_fill(
        ~attention_mask.unsqueeze(-1).bool(), 0.0
    ).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

    if embeddings is None:
        raise ValueError("Embeddings generation failed.")

    print(f"Generated embeddings of shape: {embeddings.shape}")
    return embeddings


# Function to generate embeddings
def apply_kmeans_clustering(embeddings, n_clusters=4):
    if embeddings is None:
        raise ValueError("Embeddings are None.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings.numpy())
    predicted_class = torch.tensor(clusters)
    return predicted_class


def evaluate_model(predicted_classes, true_labels):
    true_labels = true_labels.argmax(dim=1).cpu().numpy()
    predicted_classes = predicted_classes.cpu().numpy()

    accuracy = np.mean(predicted_classes == true_labels)
    f1 = f1_score(true_labels, predicted_classes, average="weighted")
    class_rep = classification_report(
        true_labels,
        predicted_classes,
        target_names=["scientific", "popular", "disinfo", "alternative_science"],
    )

    return accuracy, f1, class_rep


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("dataset", help="Path to the test data set")
    args = argparser.parse_args()

    tokenizer, model = load_model_tokenizer(BERT_MODEL_IDENTIFIER)

    # text input
    texts, labels = load_testdata(args.dataset)
    text_embeddings = generate_embeddings(tokenizer, model, texts)
    true_labels = convert_labels(labels)

    predicted_classes = apply_kmeans_clustering(text_embeddings, n_clusters=4)
    accuracy, f1, class_rep = evaluate_model(predicted_classes, true_labels)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Classification Report:\n", class_rep)
    print("done")


if __name__ == "__main__":
    main()
