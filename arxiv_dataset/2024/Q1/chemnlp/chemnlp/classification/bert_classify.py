"""Module to perform BERT classification."""
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from tqdm import tqdm
# from tqdm.notebook import tqdm
# from torchmetrics import Accuracy
# from jarvis.db.figshare import data
# import matplotlib.pyplot as plt

# df = df_cond_mat_b[0:2000]
# key = "term"
# value = "title"
# batch_size = 32
# epochs = 5

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# !export CUDA_LAUNCH_BLOCKING="1"


def train_classifier(
    df=None,
    key="categories",
    value="title_abstract",
    batch_size=32,
    epochs=5,
    seed_val=17,
    test_size=0.2,
    random_state=0,
    max_length=128,
    lr=0.001,
    model_save_frequency=3,
):
    """Train a PyTorch BERT classifier."""
    label_dict = {}
    possible_labels = df[key].unique()
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print("label_dict", label_dict)
    df["label"] = df[key].replace(label_dict)
    X_train, X_val, y_train, y_val = train_test_split(
        df.index.values,
        df.label.values,
        test_size=test_size,
        random_state=random_state,
        stratify=df.label.values,
    )
    print("X_train", X_train)
    df["data_type"] = ["not_set"] * df.shape[0]

    df.loc[X_train, "data_type"] = "train"
    df.loc[X_val, "data_type"] = "val"

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == "train"][value].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == "val"][value].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids_train = encoded_data_train["input_ids"]
    attention_masks_train = encoded_data_train["attention_mask"]
    labels_train = torch.tensor(df[df.data_type == "train"].label.values)

    input_ids_val = encoded_data_val["input_ids"]
    attention_masks_val = encoded_data_val["attention_mask"]
    labels_val = torch.tensor(df[df.data_type == "val"].label.values)

    dataset_train = TensorDataset(
        input_ids_train, attention_masks_train, labels_train
    )
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
    )

    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size,
    )

    dataloader_validation = DataLoader(
        dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=batch_size,
    )

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train) * epochs,
    )

    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average="weighted")

    def torch_accuracy_score(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, preds_flat)

    def accuracy_per_class(preds, labels):
        label_dict_inverse = {v: k for k, v in label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f"Class: {label_dict_inverse[label]}")
            print(f"Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n")

    # seed_val = 17
    # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    def evaluate(dataloader_val):
        model.to(device)
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in dataloader_val:

            batch = tuple(b.to(device) for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs["labels"].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    for epoch in tqdm(range(1, epochs + 1)):
        model.to(device)
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(
            dataloader_train,
            desc="Epoch {:1d}".format(epoch),
            leave=False,
            disable=False,
        )
        for batch in progress_bar:

            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(
                {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
            )

        if epoch == epochs:
            torch.save(
                model.state_dict(), f"finetuned_BERT_epoch_{epoch}.model"
            )

        tqdm.write(f"\nEpoch {epoch}")

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f"Training loss: {loss_train_avg}")

        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        val_acc = torch_accuracy_score(predictions, true_vals)
        tqdm.write(f"Validation loss: {val_loss}")
        tqdm.write(f"F1 Score (Weighted): {val_f1}")
        tqdm.write(f"Accuracy: {val_acc}")
        # accuracy_per_class(predictions, true_vals)


if __name__ == "__main__":
    # df = pd.read_csv("cond_mat.csv")
    df = pd.read_csv("/wrk/knc6/AtomNLP/Summarize/cond_mat.csv")
    fname = "/wrk/knc6/AtomNLP/chemnlp/chemnlp/sample_data/cond_mat_small.csv"
    # df = pd.read_csv(fname)

    train_classifier(df=df, max_length=8)
