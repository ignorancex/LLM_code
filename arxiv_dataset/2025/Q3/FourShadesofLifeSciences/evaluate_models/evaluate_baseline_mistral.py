# -*- coding: utf-8 -*-

__description__ = "evaluate non Mistral 7b models, #add your huggingface token manually"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


import argparse
from transformers import (
    BertTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch
import numpy as np
from sklearn.cluster import KMeans
import sentencepiece


MODEL_IDENTIFIER = "mistralai/Mistral-7B-v0.3"

token = '' # add your huggingface token here


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
def load_model_tokenizer(MODEL_IDENTIFIER, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER, token=token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # or load_in_4bit=True for 4-bit quantization
    )

    # Mistral 7b INIT
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_IDENTIFIER,
        num_labels=4,
        problem_type="multi_label_classification",
        token=token,
        quantization_config=quantization_config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded on device: {next(model.parameters()).device}")
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




def generate_embeddings(tokenizer, model, texts, batch_size=4):
    model.eval()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[
                -1
            ]

            # Mean pooling
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            batch_embeddings = summed / summed_mask

            embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(embeddings, dim=0)

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
    torch.cuda.is_available()
    torch.cuda.empty_cache()

    argparser = argparse.ArgumentParser()
    argparser.add_argument("dataset", help="Path to the test data set")
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text input
    texts, labels = load_testdata(args.dataset)
    tokenizer, model = load_model_tokenizer(MODEL_IDENTIFIER, device)
    tokenizer.pad_token = tokenizer.eos_token

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
