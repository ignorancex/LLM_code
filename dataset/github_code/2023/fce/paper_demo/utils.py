import os
import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from typing import Optional
from utils import *
tqdm.pandas()


####----datasetup utils----####

def preprocess(text):
    if not isinstance(text.strip(), str) or len(text.strip()) == 0:
        text = pd.NA
    return text


def strat_sampler(split, n):
    # stratified sampling on preprocessed train data
    strat_data = split.groupby('label', group_keys=False).apply(lambda x:
                                                                x.sample(int(np.rint(n * len(x) / len(split))))).sample(
        frac=1).reset_index(drop=True)

    return strat_data


def drop_empty(data_split):
    split = pd.DataFrame(data_split)
    split["text"] = split["text"].apply(preprocess)
    split = split[split["text"].notna()]

    return split


def create_data_split(data, size, ds_name, data_dir):
    data_n = DatasetDict({"train": Dataset.from_pandas(strat_sampler(drop_empty(data['train']),
                                                                     size)),
                          "test": Dataset.from_pandas(drop_empty(data['test']))})
    data_dir = data_dir
    # Check whether the specified path exists or not
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_n.save_to_disk(data_dir + "/" + ds_name + str(size) + ".hf")


####----training utils----####

class finetuning_data(LightningDataModule):
    task_field_map = {
        "ft": ["text"],
        "mia": ["predictions"]}

    task_num_labels = {
        "news": 20,
        "agnews": 4,
        "imdb": 2}

    loader_columns = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            data_dir: str = "./data",
            filename: str = "news1000.hf",
            data_name: str = "news",
            max_seq_length: int = 256,
            train_batch_size: int = 4,
            eval_batch_size: int = 4,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.data_dir = data_dir
        self.filename = filename
        self.data_name = data_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.input_fields = ["text"]
        self.num_labels = self.task_num_labels[self.data_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.original_dataset = datasets.load_from_disk(os.path.join(self.data_dir, self.filename))
        test_dataset = pd.DataFrame(self.original_dataset['test'])
        train_dataset = pd.DataFrame(self.original_dataset['train'])

        # randomly shuffle rows
        sampled = train_dataset.sample(frac=1).reset_index(drop=True)

        combined_dataset = pd.concat([train_dataset, sampled], ignore_index=True)
        combined_dataset = combined_dataset.sample(frac=1).reset_index(drop=True)  # shuffle original and augmented rows
        combined_dataset["label"] = pd.to_numeric(combined_dataset["label"])

        test_dataset = pd.DataFrame(self.original_dataset['test'])

        self.dataset = DatasetDict({"train": Dataset.from_pandas(combined_dataset),
                                    "test": Dataset.from_pandas(test_dataset)})

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]


    def prepare_data(self):
        datasets.load_from_disk(os.path.join(self.data_dir, self.filename))


    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)


    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)


    def predict_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)


    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.input_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.input_fields[0]], example_batch[self.input_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.input_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


class finetuner(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 4,
            eval_batch_size: int = 4,
            eval_splits: Optional[list] = None,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.acc = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        val_loss, logits = self(**batch)[:2]
        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        return [logits, preds, labels]

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        #         soft_preds = torch.cat([x["soft_preds"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        accuracy = self.acc.compute(predictions=preds, references=labels)["accuracy"]
        f1 = self.f1.compute(predictions=preds, references=labels, average='macro')["f1"]
        self.log_dict({"accuracy": accuracy, "f1": f1}, prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]












